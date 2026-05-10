"""
app.py — KrishiBandhu Flask Backend
Run: python app.py  →  http://localhost:5000
"""
import os, warnings
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv
import requests as req_lib
warnings.filterwarnings('ignore')
load_dotenv()

from model import (load_and_prepare, train_classifier, estimate_yield,
                   get_recommendations, CROP_INFO, YIELD_STATS, NPK_OPT,
                   FRUITS, CROPS)

app = Flask(__name__, static_folder='../frontend', static_url_path='')

@app.after_request
def cors(r):
    r.headers['Access-Control-Allow-Origin']  = '*'
    r.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    r.headers['Access-Control-Allow-Methods'] = 'GET,POST,OPTIONS'
    return r

# ── Train on startup ──────────────────────────────────────────────────────────
print("🌾 KrishiBandhu — Loading datasets and training model...")
combined_df, ds2_df = load_and_prepare()
clf, METRICS = train_classifier(combined_df)
print(f"✅ RF trained — Accuracy: {METRICS['accuracy']*100:.1f}%  CV: {METRICS['cv_mean']*100:.1f}%±{METRICS['cv_std']*100:.1f}%")
print(f"✅ {METRICS['n_crops']} crops ({METRICS['n_fruits']} fruits) | {METRICS['training_samples']} training samples")
print("🚀 Ready!\n")

# ── Static ────────────────────────────────────────────────────────────────────
@app.route('/')
def index(): return send_from_directory('../frontend', 'index.html')
@app.route('/favicon.ico')
def fav(): return '', 204

# ── /api/predict ──────────────────────────────────────────────────────────────
@app.route('/api/predict', methods=['POST','OPTIONS'])
def predict():
    if request.method == 'OPTIONS': return jsonify({}), 200
    try:
        d = request.get_json() or {}
        N   = float(d.get('N', 80));         P    = float(d.get('P', 50))
        K   = float(d.get('K', 50));         temp = float(d.get('temperature', 27))
        hum = float(d.get('humidity', 65));  ph   = float(d.get('ph', 6.5))
        rain= float(d.get('rainfall', 180)); moist= float(d.get('moisture', 27))
        ndvi= float(d.get('ndvi', 0.6))
        disease    = d.get('disease',    'None')
        fertilizer = d.get('fertilizer', 'Inorganic')
        irrigation = d.get('irrigation', 'Sprinkler')

        probs   = clf.predict_proba([[N, P, K, temp, hum, ph, rain]])[0]
        classes = clf.classes_
        all_p   = sorted(zip(classes, probs), key=lambda x: -x[1])

        crop_p  = [(c, float(p)) for c, p in all_p if c not in FRUITS]
        fruit_p = [(c, float(p)) for c, p in all_p if c in FRUITS]

        best_crop  = crop_p[0][0]  if crop_p  else None
        best_fruit = fruit_p[0][0] if fruit_p else None
        cc         = crop_p[0][1]  if crop_p  else 0
        fc         = fruit_p[0][1] if fruit_p else 0

        cy = estimate_yield(best_crop,  disease, fertilizer, irrigation, ndvi, moist, cc)  if best_crop  else {}
        fy = estimate_yield(best_fruit, disease, fertilizer, irrigation, ndvi, moist, fc)  if best_fruit else {}
        recs = get_recommendations(best_crop, N, P, K, ph, temp, rain, moist, disease, fertilizer, irrigation)

        def build_list(pairs):
            out = []
            for c, p in pairs[:5]:
                y  = estimate_yield(c, disease, fertilizer, irrigation, ndvi, moist, p)
                ci = CROP_INFO.get(c, {})
                out.append({'crop':c,'probability':round(p*100,1),
                             'yield_kg_ha':y['yield_kg_ha'],'yield_quintal_ha':y['yield_quintal_ha'],
                             'unit':y.get('unit','kg/ha'),
                             'emoji':ci.get('emoji','🌾'),'season':ci.get('season','—'),
                             'category':ci.get('category','—')})
            return out

        npk_opt = NPK_OPT.get(best_crop, {'N':(40,100),'P':(30,70),'K':(30,70)})
        npk_status = {k: {'value':v,'optimal_min':npk_opt[k][0],'optimal_max':npk_opt[k][1],
                          'status':'low' if v<npk_opt[k][0] else ('high' if v>npk_opt[k][1]+20 else 'optimal')}
                      for k, v in [('N',N),('P',P),('K',K)]}

        return jsonify({
            'success': True,
            'best_crop': best_crop,   'crop_confidence': round(cc*100,1),
            'crop_yield': cy,          'crop_info': CROP_INFO.get(best_crop,{}),
            'best_fruit': best_fruit, 'fruit_confidence': round(fc*100,1),
            'fruit_yield': fy,         'fruit_info': CROP_INFO.get(best_fruit,{}),
            'top_crops': build_list(crop_p), 'top_fruits': build_list(fruit_p),
            'recommendations': recs,   'npk_status': npk_status,
        })
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

# ── Weather helpers ───────────────────────────────────────────────────────────
STATE_WX = {
    'Maharashtra':{'temp':30,'humid':62,'wind':14,'rain':8,'desc':'Partly Cloudy','pres':1010,'vis':10},
    'Punjab':{'temp':24,'humid':55,'wind':12,'rain':2,'desc':'Sunny','pres':1012,'vis':15},
    'Rajasthan':{'temp':37,'humid':28,'wind':9,'rain':0,'desc':'Hot & Dry','pres':1008,'vis':12},
    'West Bengal':{'temp':32,'humid':82,'wind':16,'rain':18,'desc':'Humid & Rainy','pres':1006,'vis':6},
    'Uttar Pradesh':{'temp':28,'humid':65,'wind':10,'rain':5,'desc':'Overcast','pres':1009,'vis':8},
    'Madhya Pradesh':{'temp':33,'humid':50,'wind':11,'rain':3,'desc':'Warm','pres':1009,'vis':12},
    'Karnataka':{'temp':25,'humid':70,'wind':14,'rain':12,'desc':'Light Showers','pres':1011,'vis':9},
    'Tamil Nadu':{'temp':35,'humid':78,'wind':20,'rain':18,'desc':'Humid','pres':1007,'vis':7},
    'Kerala':{'temp':29,'humid':88,'wind':22,'rain':30,'desc':'Heavy Rain','pres':1005,'vis':5},
    'Andhra Pradesh':{'temp':32,'humid':68,'wind':15,'rain':10,'desc':'Partly Cloudy','pres':1009,'vis':10},
    'Telangana':{'temp':34,'humid':55,'wind':14,'rain':4,'desc':'Warm & Clear','pres':1008,'vis':11},
    'Gujarat':{'temp':34,'humid':42,'wind':16,'rain':1,'desc':'Sunny','pres':1009,'vis':14},
    'Bihar':{'temp':30,'humid':70,'wind':9,'rain':8,'desc':'Cloudy','pres':1008,'vis':8},
    'Odisha':{'temp':33,'humid':74,'wind':18,'rain':15,'desc':'Showers','pres':1007,'vis':8},
    'Haryana':{'temp':27,'humid':52,'wind':11,'rain':3,'desc':'Clear','pres':1012,'vis':15},
    'Himachal Pradesh':{'temp':18,'humid':60,'wind':8,'rain':10,'desc':'Cool & Misty','pres':1015,'vis':8},
    'Jharkhand':{'temp':30,'humid':72,'wind':10,'rain':12,'desc':'Partly Cloudy','pres':1008,'vis':9},
    'Chhattisgarh':{'temp':32,'humid':68,'wind':11,'rain':10,'desc':'Warm & Humid','pres':1008,'vis':9},
    'Assam':{'temp':28,'humid':85,'wind':12,'rain':25,'desc':'Rainy','pres':1005,'vis':6},
    'Uttarakhand':{'temp':20,'humid':65,'wind':9,'rain':15,'desc':'Cool & Cloudy','pres':1013,'vis':10},
    'Goa':{'temp':31,'humid':80,'wind':18,'rain':20,'desc':'Humid & Rainy','pres':1006,'vis':7},
}

def _sim_wx(city, state=None):
    b = STATE_WX.get(state or 'Maharashtra', STATE_WX['Maharashtra'])
    return {'success':True,'live':False,'city':city,'country':'IN',
            'temperature':b['temp'],'feels_like':b['temp']-2,'temp_min':b['temp']-3,'temp_max':b['temp']+3,
            'humidity':b['humid'],'pressure':b['pres'],'description':b['desc'],
            'wind_speed':b['wind'],'visibility':b['vis'],'clouds':40,'rain_1h':b['rain'],
            'dew_point':round(b['temp']-(100-b['humid'])/5,1)}

def _sim_fc(city):
    import random; random.seed(hash(city)%1000)
    days=['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
    base=28
    return {'success':True,'live':False,'city':city,'forecast':[
        {'date':days[i],'temp_max':base+random.randint(-3,5),'temp_min':base-random.randint(6,10),
         'description':['Sunny','Partly Cloudy','Light Rain','Cloudy','Clear','Showers','Overcast'][i%7],
         'rain':round(random.random()*12,1)} for i in range(7)]}

@app.route('/api/weather', methods=['GET'])
def weather():
    city = request.args.get('city','Nashik')
    lat  = request.args.get('lat'); lon = request.args.get('lon')
    key  = os.getenv('OPENWEATHER_API_KEY','')
    if not key or key in ('your_api_key_here','your_key_here'):
        return jsonify(_sim_wx(city))
    try:
        loc = f"lat={lat}&lon={lon}" if lat and lon else f"q={city},IN"
        r = req_lib.get(f"https://api.openweathermap.org/data/2.5/weather?{loc}&appid={key}&units=metric",timeout=8).json()
        if 'main' not in r: return jsonify(_sim_wx(city))
        t, h = round(r['main']['temp'],1), r['main']['humidity']
        return jsonify({'success':True,'live':True,'city':r.get('name',city),'country':'IN',
            'temperature':t,'feels_like':round(r['main']['feels_like'],1),
            'temp_min':round(r['main']['temp_min'],1),'temp_max':round(r['main']['temp_max'],1),
            'humidity':h,'pressure':r['main']['pressure'],
            'description':r['weather'][0]['description'].title(),
            'wind_speed':round(r.get('wind',{}).get('speed',0)*3.6,1),
            'visibility':round(r.get('visibility',10000)/1000,1),
            'clouds':r.get('clouds',{}).get('all',0),'rain_1h':r.get('rain',{}).get('1h',0),
            'dew_point':round(t-(100-h)/5,1)})
    except: return jsonify(_sim_wx(city))

@app.route('/api/forecast', methods=['GET'])
def forecast():
    city = request.args.get('city','Nashik')
    key  = os.getenv('OPENWEATHER_API_KEY','')
    if not key or key in ('your_api_key_here','your_key_here'):
        return jsonify(_sim_fc(city))
    try:
        r = req_lib.get(f"https://api.openweathermap.org/data/2.5/forecast?q={city},IN&appid={key}&units=metric&cnt=40",timeout=8).json()
        daily={}
        for item in r.get('list',[]):
            date=item['dt_txt'].split(' ')[0]
            if date not in daily: daily[date]={'temps':[],'desc':item['weather'][0]['description'],'rain':0}
            daily[date]['temps'].append(item['main']['temp']); daily[date]['rain']+=item.get('rain',{}).get('3h',0)
        fc=[{'date':d,'temp_max':round(max(i['temps']),1),'temp_min':round(min(i['temps']),1),
              'description':i['desc'].title(),'rain':round(i['rain'],1)} for d,i in list(daily.items())[:7]]
        return jsonify({'success':True,'live':True,'city':city,'forecast':fc})
    except: return jsonify(_sim_fc(city))

# ── Soil ──────────────────────────────────────────────────────────────────────
SOIL_DB = {
    'Maharashtra':  {'N':(70,120),'P':(30,60),'K':(40,80),'ph':(6.0,7.5),'moist':(20,40),'oc':(0.4,0.8),'type':'Black Cotton (Vertisols), Red Loam'},
    'Punjab':       {'N':(80,140),'P':(40,70),'K':(50,90),'ph':(7.0,8.0),'moist':(25,45),'oc':(0.6,1.0),'type':'Alluvial (Indo-Gangetic Plains)'},
    'Rajasthan':    {'N':(30,70), 'P':(20,45),'K':(60,110),'ph':(7.5,9.0),'moist':(10,25),'oc':(0.2,0.5),'type':'Arid Sandy Desert Soil'},
    'West Bengal':  {'N':(60,100),'P':(25,55),'K':(35,65),'ph':(5.5,6.5),'moist':(40,70),'oc':(0.7,1.2),'type':'Alluvial, Laterite'},
    'Karnataka':    {'N':(55,100),'P':(30,60),'K':(45,85),'ph':(5.8,7.0),'moist':(25,50),'oc':(0.5,0.9),'type':'Red Sandy Loam, Black Cotton'},
    'Tamil Nadu':   {'N':(50,90), 'P':(25,55),'K':(40,80),'ph':(6.0,7.5),'moist':(30,55),'oc':(0.4,0.8),'type':'Red Loam, Alluvial'},
    'Uttar Pradesh':{'N':(70,130),'P':(35,65),'K':(50,90),'ph':(6.5,8.0),'moist':(30,50),'oc':(0.5,0.9),'type':'Alluvial (Gangetic)'},
    'Gujarat':      {'N':(40,80), 'P':(20,50),'K':(55,100),'ph':(7.0,8.5),'moist':(15,35),'oc':(0.3,0.6),'type':'Alluvial, Black, Sandy Loam'},
    'Andhra Pradesh':{'N':(55,95),'P':(28,58),'K':(40,80),'ph':(6.0,7.5),'moist':(25,50),'oc':(0.4,0.8),'type':'Red, Black, Alluvial'},
    'Kerala':       {'N':(45,85), 'P':(20,50),'K':(50,100),'ph':(4.5,6.0),'moist':(50,80),'oc':(1.0,2.0),'type':'Laterite, Alluvial'},
    'Madhya Pradesh':{'N':(55,100),'P':(25,55),'K':(45,85),'ph':(6.0,7.5),'moist':(22,42),'oc':(0.4,0.8),'type':'Black Cotton, Red, Alluvial'},
    'Haryana':      {'N':(75,130),'P':(35,65),'K':(45,85),'ph':(7.0,8.5),'moist':(25,45),'oc':(0.4,0.8),'type':'Alluvial Sandy Loam'},
    'Bihar':        {'N':(65,110),'P':(30,60),'K':(45,85),'ph':(6.5,7.8),'moist':(30,55),'oc':(0.5,0.9),'type':'Alluvial (Gangetic), Terai'},
    'Odisha':       {'N':(55,95), 'P':(25,55),'K':(40,80),'ph':(5.5,6.8),'moist':(35,60),'oc':(0.5,1.0),'type':'Red Laterite, Alluvial'},
    'Telangana':    {'N':(50,90), 'P':(25,55),'K':(40,80),'ph':(6.0,7.5),'moist':(22,45),'oc':(0.4,0.8),'type':'Red Sandy, Black Cotton'},
    'Himachal Pradesh':{'N':(60,100),'P':(30,65),'K':(50,90),'ph':(5.5,7.0),'moist':(30,55),'oc':(0.8,1.5),'type':'Hill Soil, Brown Forest'},
    'Jharkhand':    {'N':(45,85), 'P':(20,50),'K':(40,75),'ph':(4.5,6.5),'moist':(25,50),'oc':(0.5,1.0),'type':'Red Laterite, Sandy Loam'},
    'Uttarakhand':  {'N':(55,95), 'P':(25,55),'K':(45,80),'ph':(5.5,7.0),'moist':(30,55),'oc':(0.7,1.3),'type':'Forest, Alluvial'},
    'Chhattisgarh': {'N':(50,90), 'P':(22,52),'K':(40,75),'ph':(5.5,7.0),'moist':(28,50),'oc':(0.4,0.9),'type':'Red, Yellow, Alluvial'},
    'Assam':        {'N':(55,95), 'P':(25,55),'K':(40,80),'ph':(4.5,6.5),'moist':(45,75),'oc':(0.8,1.5),'type':'Alluvial, Red Laterite'},
    'Goa':          {'N':(45,85), 'P':(20,50),'K':(40,80),'ph':(4.5,6.5),'moist':(45,75),'oc':(0.6,1.2),'type':'Laterite, Alluvial Coastal'},
}

@app.route('/api/soil', methods=['GET'])
def soil():
    state = request.args.get('state','Maharashtra')
    city  = request.args.get('city','Nashik')
    p = SOIL_DB.get(state, SOIL_DB['Maharashtra'])
    rng = np.random.RandomState(hash(city+state) % (2**32))
    N   = round(float(rng.uniform(*p['N'])),1)
    P   = round(float(rng.uniform(*p['P'])),1)
    K   = round(float(rng.uniform(*p['K'])),1)
    ph  = round(float(rng.uniform(*p['ph'])),2)
    moist = round(float(rng.uniform(*p['moist'])),1)
    oc    = round(float(rng.uniform(*p['oc'])),2)
    ec    = round(float(rng.uniform(0.2,1.2)),2)
    score = max(18, min(97, round(
        max(0,1-abs(ph-6.8)/2)*28 + min(1,N/120)*24 +
        min(1,P/60)*18 + min(1,K/80)*16 + min(1,oc/1.2)*14)))
    grade = 'Excellent' if score>80 else 'Good' if score>65 else 'Fair' if score>45 else 'Poor'
    recs  = []
    if N < p['N'][0]: recs.append({'icon':'🌿','text':f'Apply Urea — Nitrogen deficient ({N} kg/ha)'})
    if P < p['P'][0]: recs.append({'icon':'🔴','text':'Apply DAP — Phosphorus is low'})
    if K < p['K'][0]: recs.append({'icon':'🟡','text':'Apply MOP — Potassium boost needed'})
    if ph < 6.0: recs.append({'icon':'🪨','text':f'Apply lime — soil is acidic (pH {ph})'})
    elif ph > 7.8: recs.append({'icon':'🧂','text':f'Apply gypsum — soil is alkaline (pH {ph})'})
    recs.append({'icon':'🌱','text':'Add FYM/Vermicompost @ 5–10 t/ha to improve organic carbon'})
    return jsonify({'success':True,'state':state,'city':city,
                    'npk':{'N':N,'P':P,'K':K},'ph':ph,'moisture':moist,
                    'organic_carbon':oc,'electrical_conductivity':ec,'soil_type':p['type'],
                    'health_score':score,'health_grade':grade,'recommendations':recs[:5]})

# ── Model info ────────────────────────────────────────────────────────────────
@app.route('/api/model-info', methods=['GET'])
def model_info():
    return jsonify({'success':True,'classifier':{
        'type':'RandomForestClassifier','n_estimators':METRICS['n_estimators'],
        'accuracy':METRICS['accuracy'],'accuracy_pct':round(METRICS['accuracy']*100,1),
        'cv_mean_pct':round(METRICS['cv_mean']*100,1),'cv_std_pct':round(METRICS['cv_std']*100,1),
        'training_samples':METRICS['training_samples'],'n_crops':METRICS['n_crops'],
        'n_fruits':METRICS['n_fruits'],'feature_importance':METRICS['feature_importance'],
        'classes':METRICS['classes']},
        'datasets':{'ds3':{'name':'Indian Crop Dataset','rows':2200,'crops':22},
                    'synthetic':{'rows':400,'crops':4},'combined':{'rows':2600,'crops':26,'fruits':10}}})

@app.route('/api/crops', methods=['GET'])
def crops():
    return jsonify({'success':True,'crops':{
        c:{**i,'yield_stats':YIELD_STATS.get(c,{}),'npk_optimal':NPK_OPT.get(c,{})}
        for c,i in CROP_INFO.items()}})

# ── States / Cities ───────────────────────────────────────────────────────────
STATES = {
    "Andhra Pradesh":["Vijayawada","Visakhapatnam","Guntur","Nellore","Kurnool","Tirupati","Rajahmundry","Kadapa"],
    "Assam":["Guwahati","Silchar","Dibrugarh","Jorhat","Nagaon","Tinsukia","Tezpur"],
    "Bihar":["Patna","Gaya","Muzaffarpur","Bhagalpur","Darbhanga","Purnia","Arrah","Begusarai"],
    "Chhattisgarh":["Raipur","Bhilai","Bilaspur","Durg","Korba","Rajnandgaon"],
    "Goa":["Panaji","Margao","Vasco da Gama","Mapusa","Ponda"],
    "Gujarat":["Ahmedabad","Surat","Vadodara","Rajkot","Bhavnagar","Jamnagar","Junagadh","Gandhinagar","Anand","Navsari"],
    "Haryana":["Faridabad","Gurgaon","Panipat","Ambala","Yamunanagar","Rohtak","Hisar","Karnal","Sonipat"],
    "Himachal Pradesh":["Shimla","Mandi","Solan","Dharamsala","Baddi","Palampur","Kullu"],
    "Jharkhand":["Ranchi","Jamshedpur","Dhanbad","Bokaro","Deoghar","Hazaribagh"],
    "Karnataka":["Bengaluru","Mysuru","Hubballi","Mangaluru","Belagavi","Kalaburagi","Ballari","Vijayapura","Shivamogga","Tumkur","Davangere"],
    "Kerala":["Thiruvananthapuram","Kochi","Kozhikode","Thrissur","Kannur","Kollam","Palakkad","Alappuzha"],
    "Madhya Pradesh":["Indore","Bhopal","Jabalpur","Gwalior","Ujjain","Sagar","Dewas","Satna","Ratlam"],
    "Maharashtra":["Mumbai","Pune","Nagpur","Nashik","Aurangabad","Solapur","Amravati","Kolhapur","Nanded","Jalgaon","Akola","Latur","Dhule","Ahmednagar","Chandrapur"],
    "Odisha":["Bhubaneswar","Cuttack","Rourkela","Brahmapur","Sambalpur","Puri","Balasore"],
    "Punjab":["Ludhiana","Amritsar","Jalandhar","Patiala","Bathinda","Mohali","Hoshiarpur","Pathankot","Moga"],
    "Rajasthan":["Jaipur","Jodhpur","Kota","Bikaner","Ajmer","Udaipur","Bhilwara","Alwar","Bharatpur","Sikar","Sri Ganganagar"],
    "Tamil Nadu":["Chennai","Coimbatore","Madurai","Tiruchirappalli","Salem","Tirunelveli","Tiruppur","Erode","Vellore","Thanjavur"],
    "Telangana":["Hyderabad","Warangal","Nizamabad","Khammam","Karimnagar","Ramagundam","Nalgonda"],
    "Uttar Pradesh":["Lucknow","Kanpur","Agra","Varanasi","Prayagraj","Meerut","Noida","Ghaziabad","Bareilly","Aligarh","Gorakhpur"],
    "Uttarakhand":["Dehradun","Haridwar","Roorkee","Haldwani","Rudrapur","Kashipur","Rishikesh"],
    "West Bengal":["Kolkata","Asansol","Siliguri","Durgapur","Bardhaman","Malda","Barasat","Kharagpur"],
}

@app.route('/api/states', methods=['GET'])
def states(): return jsonify({'success':True,'states':list(STATES.keys())})

@app.route('/api/cities/<state>', methods=['GET'])
def cities(state): return jsonify({'success':True,'state':state,'cities':STATES.get(state,[])})

# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n" + "="*55)
    print("  🌾  KrishiBandhu — AI Crop Prediction System")
    print("="*55)
    print("  http://localhost:5000")
    print("="*55+"\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
