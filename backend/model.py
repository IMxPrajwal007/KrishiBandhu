"""
model.py — KrishiBandhu ML Model
Primary: indiancrop_dataset.csv  (2200 rows, 22 crops incl. 10 fruits)
Augment: +400 synthetic rows for Wheat, Soybean, Barley, Sugarcane → 26 crops total
Returns BOTH best crop (field/legume) AND best fruit predictions
"""
import os, warnings
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
warnings.filterwarnings('ignore')

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

FRUITS   = ['Apple','Banana','Grapes','Mango','Muskmelon','Orange','Papaya','Pomegranate','Watermelon','Coconut']
LEGUMES  = ['ChickPea','KidneyBeans','PigeonPeas','MothBeans','MungBean','Blackgram','Lentil']
FIELD    = ['Rice','Wheat','Maize','Cotton','Jute','Coffee','Soybean','Barley','Sugarcane']
CROPS    = FIELD + LEGUMES   # non-fruit crops

YIELD_STATS = {
    'Rice':        {'mean':3896,  'std':1150,  'min':2067,  'max':5953,  'unit':'kg/ha'},
    'Wheat':       {'mean':4078,  'std':1123,  'min':2043,  'max':5894,  'unit':'kg/ha'},
    'Maize':       {'mean':3983,  'std':1212,  'min':2024,  'max':5953,  'unit':'kg/ha'},
    'Cotton':      {'mean':3926,  'std':1201,  'min':2046,  'max':5981,  'unit':'kg/ha'},
    'Soybean':     {'mean':4257,  'std':1156,  'min':2029,  'max':5998,  'unit':'kg/ha'},
    'Barley':      {'mean':3800,  'std':1100,  'min':1800,  'max':5500,  'unit':'kg/ha'},
    'Sugarcane':   {'mean':65000, 'std':15000, 'min':40000, 'max':90000, 'unit':'kg/ha'},
    'Jute':        {'mean':2500,  'std':600,   'min':1500,  'max':3800,  'unit':'kg/ha'},
    'Coffee':      {'mean':800,   'std':200,   'min':400,   'max':1400,  'unit':'kg/ha'},
    'ChickPea':    {'mean':1200,  'std':300,   'min':600,   'max':2000,  'unit':'kg/ha'},
    'KidneyBeans': {'mean':1400,  'std':350,   'min':700,   'max':2200,  'unit':'kg/ha'},
    'PigeonPeas':  {'mean':1100,  'std':280,   'min':500,   'max':1800,  'unit':'kg/ha'},
    'MothBeans':   {'mean':900,   'std':220,   'min':400,   'max':1500,  'unit':'kg/ha'},
    'MungBean':    {'mean':1000,  'std':250,   'min':500,   'max':1700,  'unit':'kg/ha'},
    'Blackgram':   {'mean':950,   'std':240,   'min':450,   'max':1600,  'unit':'kg/ha'},
    'Lentil':      {'mean':1150,  'std':290,   'min':550,   'max':1900,  'unit':'kg/ha'},
    'Mango':       {'mean':12000, 'std':4000,  'min':5000,  'max':25000, 'unit':'kg/ha'},
    'Banana':      {'mean':35000, 'std':10000, 'min':18000, 'max':60000, 'unit':'kg/ha'},
    'Grapes':      {'mean':20000, 'std':6000,  'min':8000,  'max':40000, 'unit':'kg/ha'},
    'Apple':       {'mean':15000, 'std':5000,  'min':6000,  'max':30000, 'unit':'kg/ha'},
    'Orange':      {'mean':18000, 'std':5500,  'min':8000,  'max':35000, 'unit':'kg/ha'},
    'Papaya':      {'mean':40000, 'std':12000, 'min':20000, 'max':75000, 'unit':'kg/ha'},
    'Pomegranate': {'mean':14000, 'std':4000,  'min':6000,  'max':25000, 'unit':'kg/ha'},
    'Watermelon':  {'mean':25000, 'std':7000,  'min':12000, 'max':45000, 'unit':'kg/ha'},
    'Muskmelon':   {'mean':18000, 'std':5000,  'min':8000,  'max':32000, 'unit':'kg/ha'},
    'Coconut':     {'mean':9000,  'std':2500,  'min':4000,  'max':16000, 'unit':'nuts/ha'},
}

DISEASE_FX    = {'None':1.00,'Mild':0.92,'Moderate':0.86,'Severe':0.74}
FERTILIZER_FX = {'Organic':1.02,'Inorganic':1.04,'Mixed':0.99}
IRRIGATION_FX = {'Drip':1.04,'Sprinkler':1.03,'Manual':1.01,'None':0.97}

NPK_OPT = {
    'Rice':        {'N':(60,99),   'P':(35,60),  'K':(35,45)},
    'Maize':       {'N':(60,100),  'P':(35,60),  'K':(15,25)},
    'Cotton':      {'N':(100,140), 'P':(35,60),  'K':(15,25)},
    'Jute':        {'N':(60,100),  'P':(35,60),  'K':(35,45)},
    'Coffee':      {'N':(80,120),  'P':(15,40),  'K':(25,35)},
    'Wheat':       {'N':(100,140), 'P':(50,70),  'K':(40,60)},
    'Soybean':     {'N':(20,60),   'P':(60,100), 'K':(60,100)},
    'Barley':      {'N':(60,100),  'P':(40,70),  'K':(60,100)},
    'Sugarcane':   {'N':(100,150), 'P':(50,80),  'K':(80,120)},
    'ChickPea':    {'N':(20,60),   'P':(55,80),  'K':(75,85)},
    'KidneyBeans': {'N':(0,40),    'P':(55,80),  'K':(15,25)},
    'PigeonPeas':  {'N':(0,40),    'P':(55,80),  'K':(15,25)},
    'MothBeans':   {'N':(0,40),    'P':(35,60),  'K':(15,25)},
    'MungBean':    {'N':(0,40),    'P':(35,60),  'K':(15,25)},
    'Blackgram':   {'N':(20,60),   'P':(55,80),  'K':(15,25)},
    'Lentil':      {'N':(0,40),    'P':(55,80),  'K':(15,25)},
    'Apple':       {'N':(0,40),    'P':(120,145),'K':(195,205)},
    'Banana':      {'N':(80,120),  'P':(70,95),  'K':(45,55)},
    'Grapes':      {'N':(0,40),    'P':(120,145),'K':(195,205)},
    'Mango':       {'N':(0,40),    'P':(15,40),  'K':(25,35)},
    'Muskmelon':   {'N':(80,120),  'P':(5,30),   'K':(45,55)},
    'Orange':      {'N':(0,40),    'P':(5,30),   'K':(5,15)},
    'Papaya':      {'N':(31,70),   'P':(46,70),  'K':(45,55)},
    'Pomegranate': {'N':(0,40),    'P':(5,30),   'K':(35,45)},
    'Watermelon':  {'N':(80,120),  'P':(5,30),   'K':(45,55)},
    'Coconut':     {'N':(0,40),    'P':(5,30),   'K':(25,35)},
}

CROP_INFO = {
    'Rice':        {'emoji':'🌾','category':'Field Crop','season':'Kharif (Jun–Oct)','water':'High (1000–2500 mm)','ph_range':'5.5–7.0','temp_range':'20–35°C','soil':'Clay loam, flooded paddies','fertilizer_tips':['Urea 120 kg/ha','DAP 60 kg/ha','MOP 40 kg/ha','Zinc Sulfate 25 kg/ha'],'tips':['Maintain 5–10 cm standing water during vegetative stage','Apply nitrogen in 3 splits: basal, tillering, panicle initiation','Use certified disease-free seeds to prevent blast','Drain field 2 weeks before harvest']},
    'Wheat':       {'emoji':'🌾','category':'Field Crop','season':'Rabi (Nov–Mar)','water':'Moderate (300–800 mm)','ph_range':'6.0–7.5','temp_range':'10–25°C','soil':'Well-drained loam, clay loam','fertilizer_tips':['Urea 150 kg/ha (split)','SSP 80 kg/ha','MOP 60 kg/ha','FYM 10 t/ha'],'tips':['Sow in October–November for best yield','First irrigation at crown root initiation (21 days)','Apply second N dose at first node stage','Watch for yellow rust — spray Propiconazole']},
    'Maize':       {'emoji':'🌽','category':'Field Crop','season':'Kharif + Rabi','water':'Moderate (500–1200 mm)','ph_range':'5.5–7.5','temp_range':'18–35°C','soil':'Sandy loam to clay loam','fertilizer_tips':['Urea 180 kg/ha','DAP 75 kg/ha','MOP 40 kg/ha','Zinc Sulfate 25 kg/ha'],'tips':['Use hybrid seeds for 40–60% higher yield','Ensure adequate zinc — maize is highly zinc-sensitive','Critical irrigation at tasseling and grain filling','Fall Armyworm (FAW) is a major pest — monitor early']},
    'Cotton':      {'emoji':'🌸','category':'Field Crop','season':'Kharif (May–Nov)','water':'Moderate (500–1200 mm)','ph_range':'6.0–8.0','temp_range':'21–38°C','soil':'Black cotton soil (Vertisols)','fertilizer_tips':['Urea 120 kg/ha','DAP 60 kg/ha','MOP 60 kg/ha','Sulfur 20 kg/ha'],'tips':['Bt Cotton recommended for bollworm resistance','Apply Mepiquat Chloride to control vegetative growth','Monitor for Pink Bollworm — use pheromone traps','Avoid waterlogging on black soils']},
    'Soybean':     {'emoji':'🫘','category':'Field Crop','season':'Kharif (Jun–Sep)','water':'Moderate (600–1200 mm)','ph_range':'6.0–7.5','temp_range':'20–32°C','soil':'Well-drained loam, medium black','fertilizer_tips':['Rhizobium inoculant (seed treatment)','SSP 50 kg/ha','MOP 40 kg/ha','Sulfur 20 kg/ha'],'tips':['Use Rhizobium inoculant — soybean fixes its own nitrogen','Avoid excess nitrogen fertilizer','Critical water need at flowering and pod filling','Yellow mosaic virus — spread by whitefly, control early']},
    'Barley':      {'emoji':'🌾','category':'Field Crop','season':'Rabi (Oct–Mar)','water':'Low (250–500 mm)','ph_range':'6.0–8.0','temp_range':'12–25°C','soil':'Sandy loam, light loam','fertilizer_tips':['Urea 80 kg/ha','SSP 60 kg/ha','MOP 40 kg/ha','FYM 8 t/ha'],'tips':['Most drought-tolerant cereal — good for low rainfall areas','Avoid late sowing after November','Apply single N dose at sowing for short varieties','Stripe rust is the key disease — use resistant varieties']},
    'Sugarcane':   {'emoji':'🎋','category':'Field Crop','season':'Year-round (plant Oct–Nov)','water':'High (1500–2500 mm)','ph_range':'6.0–8.0','temp_range':'20–38°C','soil':'Deep loam, alluvial','fertilizer_tips':['Urea 250 kg/ha (3 splits)','SSP 100 kg/ha','MOP 100 kg/ha','FYM 20 t/ha'],'tips':['Drip irrigation saves 30–40% water','Apply trash mulching to conserve moisture','Ratoon crop gives 70–80% yield at lower cost','Monitor for Red Rot — most serious disease']},
    'Jute':        {'emoji':'🌿','category':'Field Crop','season':'Kharif (Mar–Jun)','water':'High (1200–2000 mm)','ph_range':'5.5–7.5','temp_range':'23–37°C','soil':'Loamy, alluvial, river basins','fertilizer_tips':['Urea 60 kg/ha','SSP 40 kg/ha','MOP 30 kg/ha','FYM 5 t/ha'],'tips':['West Bengal and Assam are top producers','Retting in clean, slow-moving water gives quality fibre','Harvest at 50% flowering for best fibre quality','Stem rot is the main disease — ensure good drainage']},
    'Coffee':      {'emoji':'☕','category':'Plantation Crop','season':'Perennial (harvest Oct–Feb)','water':'Moderate (1200–2200 mm)','ph_range':'5.5–6.5','temp_range':'15–30°C','soil':'Well-drained laterite, loam','fertilizer_tips':['Urea 200 g/plant','SSP 200 g/plant','MOP 200 g/plant','Borax 5 g/plant'],'tips':['Karnataka produces 70% of Indian coffee','Shade trees like silver oak improve quality','Coffee Berry Borer is the most destructive pest','Pulping and fermentation quality determines export grade']},
    'ChickPea':    {'emoji':'🟡','category':'Legume','season':'Rabi (Oct–Mar)','water':'Low (350–500 mm)','ph_range':'6.0–8.5','temp_range':'10–30°C','soil':'Well-drained sandy loam to clay','fertilizer_tips':['Rhizobium inoculant','SSP 60 kg/ha','MOP 30 kg/ha','Sulfur 20 kg/ha'],'tips':['India is world\'s largest producer and consumer','Fusarium wilt is major — use resistant varieties','Avoid excess moisture — susceptible to waterlogging','Pod borer (Helicoverpa) is the key pest']},
    'KidneyBeans': {'emoji':'🫘','category':'Legume','season':'Kharif (Jun–Sep)','water':'Moderate (300–600 mm)','ph_range':'6.0–7.5','temp_range':'15–25°C','soil':'Well-drained fertile loam','fertilizer_tips':['Rhizobium inoculant','DAP 50 kg/ha','MOP 40 kg/ha','Zinc 20 kg/ha'],'tips':['Jammu & Kashmir and HP are top producers','Frost-sensitive — plant after last frost','White mold is a major disease in humid conditions','Trellising improves yield and air circulation']},
    'PigeonPeas':  {'emoji':'🫛','category':'Legume','season':'Kharif (Jun–Nov)','water':'Low (600–1000 mm)','ph_range':'5.5–7.5','temp_range':'18–38°C','soil':'Sandy loam to clay loam','fertilizer_tips':['Rhizobium inoculant','SSP 50 kg/ha','MOP 30 kg/ha'],'tips':['Excellent drought tolerance once established','Intercropping with cereals gives higher output','Watch for wilt and sterility mosaic diseases','Maharashtra and Karnataka are top producers']},
    'MothBeans':   {'emoji':'🌱','category':'Legume','season':'Kharif (Jun–Sep)','water':'Very Low (200–400 mm)','ph_range':'6.5–8.5','temp_range':'24–38°C','soil':'Sandy, light loam — arid zones','fertilizer_tips':['Rhizobium inoculant','SSP 30 kg/ha','MOP 20 kg/ha'],'tips':['Most drought-resistant Indian legume','Rajasthan is the top producer','Suited to arid and semi-arid zones','Good candidate for crop diversification in dry areas']},
    'MungBean':    {'emoji':'🌿','category':'Legume','season':'Kharif (Jun–Sep)','water':'Low (400–600 mm)','ph_range':'6.2–7.2','temp_range':'28–35°C','soil':'Well-drained sandy loam','fertilizer_tips':['Rhizobium inoculant','SSP 40 kg/ha','MOP 30 kg/ha'],'tips':['Short duration (60–90 days) — ideal catch crop','Susceptible to excessive moisture','Mung bean yellow mosaic virus (MYMV) is key disease','Aphids and jassids are major pests']},
    'Blackgram':   {'emoji':'⚫','category':'Legume','season':'Kharif + Rabi','water':'Low (350–550 mm)','ph_range':'5.5–7.5','temp_range':'25–35°C','soil':'Sandy loam, black cotton','fertilizer_tips':['Rhizobium inoculant','SSP 40 kg/ha','MOP 25 kg/ha'],'tips':['High protein urad dal crop','Warm, humid conditions needed for germination','Powdery mildew and leaf curl are major diseases','Harvest when 80% pods turn black']},
    'Lentil':      {'emoji':'🟤','category':'Legume','season':'Rabi (Oct–Mar)','water':'Low (250–400 mm)','ph_range':'6.0–8.0','temp_range':'18–30°C','soil':'Sandy loam to clay loam','fertilizer_tips':['Rhizobium inoculant','SSP 50 kg/ha','MOP 25 kg/ha'],'tips':['Madhya Pradesh and UP are top producers','Frost-tolerant during vegetative stage','Collar rot and rust are major diseases','Harvest when lower pods turn yellow-brown']},
    'Mango':       {'emoji':'🥭','category':'Fruit','season':'Summer (Mar–Jun)','water':'Moderate (750–1500 mm)','ph_range':'5.5–7.5','temp_range':'24–38°C','soil':'Deep alluvial loam','fertilizer_tips':['NPK 200:100:200 g/tree','FYM 20 kg/tree','Boron at flowering','Micronutrients foliar spray'],'tips':['National fruit of India — Alphonso, Kesar, Dasheri are premium varieties','Dry period essential for flower induction','Mango hopper is the major pest — spray at budding','India produces 50% of world mango supply']},
    'Banana':      {'emoji':'🍌','category':'Fruit','season':'Year-round (10–14 months)','water':'High (1200–2200 mm)','ph_range':'6.0–7.5','temp_range':'15–35°C','soil':'Deep loam, rich fertile','fertilizer_tips':['Drip fertigation NPK','Calcium + Magnesium','Micronutrients monthly','FYM 20 kg/plant'],'tips':['Jalgaon is the Banana Capital of India','Grand Naine dominates commercial cultivation','Panama wilt — use disease-free suckers','Drip fertigation is essential for commercial production']},
    'Grapes':      {'emoji':'🍇','category':'Fruit','season':'Winter harvest (Dec–Mar)','water':'Moderate (700–1200 mm)','ph_range':'6.5–7.5','temp_range':'15–40°C','soil':'Sandy loam, well-drained','fertilizer_tips':['Calcium Nitrate via drip','Potassium Nitrate at veraison','K2SO4 15 g/vine','Zinc + Boron foliar spray'],'tips':['Nashik is the Wine Capital — 75% of Indian wine production','Thompson Seedless and Sonaka for export market','Powdery mildew is the key disease','Pruning management determines yield and quality']},
    'Apple':       {'emoji':'🍎','category':'Fruit','season':'Summer (Jul–Oct)','water':'Moderate (1000–1500 mm)','ph_range':'5.5–7.0','temp_range':'20–24°C','soil':'Well-drained loam, alluvial','fertilizer_tips':['Urea 1.5 kg/tree','SSP 2 kg/tree','MOP 2 kg/tree','Calcium Nitrate foliar spray'],'tips':['Himachal Pradesh produces 90% of Indian apples','Chilling requirement: 1000–1200 hours below 7°C','Scab and woolly aphid are major problems','High-density planting (HDPS) increases productivity 3x']},
    'Orange':      {'emoji':'🍊','category':'Fruit','season':'Winter (Nov–Mar)','water':'Moderate (750–1250 mm)','ph_range':'6.0–7.5','temp_range':'15–30°C','soil':'Sandy loam, laterite','fertilizer_tips':['Urea 1 kg/tree','SSP 1.5 kg/tree','MOP 1 kg/tree','Micronutrient cocktail spray'],'tips':['Nagpur Orange is the most famous Indian variety','Citrus decline — manage nutrition carefully','Greening disease (HLB) is the most destructive','Drip fertigation improves fruit size and juice content']},
    'Papaya':      {'emoji':'🍈','category':'Fruit','season':'Year-round (9–12 months)','water':'Moderate (1000–1500 mm)','ph_range':'6.0–7.0','temp_range':'22–38°C','soil':'Sandy loam, well-drained','fertilizer_tips':['Urea 200 g/plant/month','SSP 250 g/plant','MOP 200 g/plant','FYM 10 kg/plant'],'tips':['AP and Gujarat are top producers','Papaya ringspot virus (PRSV) is the biggest threat','Avoid waterlogging — very sensitive to root rot','Harvest when 10–15% of fruit surface turns yellow']},
    'Pomegranate': {'emoji':'❤️','category':'Fruit','season':'Year-round (harvest Feb–May)','water':'Low–Moderate (500–800 mm)','ph_range':'5.5–7.2','temp_range':'25–38°C','soil':'Sandy loam to clay','fertilizer_tips':['NPK balanced 600:300:300 g/plant','K2SO4 at fruit development','Boron + Calcium foliar spray','FYM 20 kg/plant'],'tips':['Solapur Bhagwa variety exported to Europe','Drought-tolerant once established','Bacterial blight (Xanthomonas) is the key disease','Bagging fruits improves colour and prevents pests']},
    'Watermelon':  {'emoji':'🍉','category':'Fruit','season':'Summer (Feb–May)','water':'Moderate (400–600 mm)','ph_range':'6.0–7.0','temp_range':'24–35°C','soil':'Sandy loam, well-drained','fertilizer_tips':['Urea 80 kg/ha','SSP 60 kg/ha','MOP 60 kg/ha','Boron 2 kg/ha'],'tips':['AP and Karnataka are top producers','Seedless varieties command premium prices','Downy mildew and Fusarium wilt are major diseases','Drip irrigation with black mulch maximizes yields']},
    'Muskmelon':   {'emoji':'🍈','category':'Fruit','season':'Summer (Mar–Jun)','water':'Low–Moderate (300–500 mm)','ph_range':'6.0–7.0','temp_range':'25–35°C','soil':'Sandy loam, light soil','fertilizer_tips':['Urea 60 kg/ha','SSP 50 kg/ha','MOP 50 kg/ha','Micronutrients foliar spray'],'tips':['Harvest at slip stage when fruit separates from vine','Powdery mildew is the common disease','Short duration — ideal for crop rotation','Net covering prevents insect pests']},
    'Coconut':     {'emoji':'🥥','category':'Fruit/Plantation','season':'Year-round (perennial)','water':'High (1000–2500 mm)','ph_range':'5.5–8.0','temp_range':'27–32°C','soil':'Sandy loam to laterite, coastal','fertilizer_tips':['Urea 1.3 kg/palm/year','SSP 2 kg/palm/year','MOP 3.5 kg/palm/year','FYM 50 kg/palm'],'tips':['Kerala is the Coconut Capital — Kera means coconut tree','Rhinoceros beetle is the most destructive pest','Eriophyid mite causes mite wilt — apply wettable sulfur','Intercropping with cocoa and banana is highly profitable']},
}


def load_and_prepare():
    """Load DS3 + synthetic rows for 4 missing crops. Returns (combined_df, ds2_df)."""
    df3 = pd.read_csv(os.path.join(DATA_DIR, 'indiancrop_dataset.csv'))
    df3.columns = df3.columns.str.strip()
    df3 = df3.rename(columns={'N_SOIL':'N','P_SOIL':'P','K_SOIL':'K',
                               'TEMPERATURE':'temp','HUMIDITY':'humid','RAINFALL':'rain','CROP':'crop'})
    np.random.seed(99)
    extra = []
    specs = {
        'Wheat':     {'N':(100,140),'P':(50,70), 'K':(40,60), 'temp':(10,25),'humid':(55,75),'ph':(6.0,7.5),'rain':(300,800)},
        'Soybean':   {'N':(20,60),  'P':(60,100),'K':(60,100),'temp':(20,32),'humid':(60,85),'ph':(6.0,7.5),'rain':(600,1200)},
        'Barley':    {'N':(60,100), 'P':(40,70), 'K':(60,100),'temp':(12,25),'humid':(50,70),'ph':(6.0,8.0),'rain':(250,500)},
        'Sugarcane': {'N':(100,150),'P':(50,80), 'K':(80,120),'temp':(22,38),'humid':(60,85),'ph':(6.0,8.0),'rain':(400,800)},
    }
    for crop, s in specs.items():
        for _ in range(100):
            extra.append({k: float(np.random.uniform(*v)) for k, v in s.items()} | {'crop': crop})
    feats = ['N','P','K','temp','humid','ph','rain']
    combined = pd.concat([df3[feats+['crop']], pd.DataFrame(extra)[feats+['crop']]], ignore_index=True)

    ds2 = pd.read_csv(os.path.join(DATA_DIR, 'Smart_Farming_Crop_Yield_2024.csv'))
    ds2.columns = ds2.columns.str.strip()
    ds2 = ds2.rename(columns={'crop_type':'crop','soil_pH':'ph','temperature_C':'temp',
                               'rainfall_mm':'rain','humidity_%':'humid','soil_moisture_%':'moisture',
                               'yield_kg_per_hectare':'yield','crop_disease_status':'disease',
                               'fertilizer_type':'fertilizer','irrigation_type':'irrigation'})
    return combined, ds2


def train_classifier(df):
    feats = ['N','P','K','temp','humid','ph','rain']
    X, y = df[feats], df['crop']
    Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf = RandomForestClassifier(n_estimators=300, max_depth=15, min_samples_leaf=2,
                                  random_state=42, n_jobs=-1)
    clf.fit(Xt, yt)
    acc   = accuracy_score(ye, clf.predict(Xe))
    cv    = cross_val_score(clf, X, y, cv=5)
    fi    = dict(zip(feats, clf.feature_importances_.tolist()))
    return clf, {
        'accuracy': round(float(acc), 4),
        'cv_mean':  round(float(cv.mean()), 4),
        'cv_std':   round(float(cv.std()), 4),
        'feature_importance': {k: round(v,4) for k,v in sorted(fi.items(), key=lambda x:-x[1])},
        'classes': clf.classes_.tolist(),
        'n_estimators': clf.n_estimators,
        'training_samples': len(Xt),
        'test_samples': len(Xe),
        'n_crops': len(clf.classes_),
        'n_fruits': len([c for c in clf.classes_ if c in FRUITS]),
    }


def estimate_yield(crop, disease='None', fertilizer='Inorganic',
                   irrigation='Sprinkler', ndvi=0.6, moisture=27.0, confidence=0.5):
    s = YIELD_STATS.get(crop, {'mean':3000,'std':800,'min':1000,'max':6000,'unit':'kg/ha'})
    base  = s['mean'] + (confidence - 0.5) * s['std'] * 1.5
    base  = max(s['min'], min(s['max'], base))
    final = base * DISEASE_FX.get(disease,1) * FERTILIZER_FX.get(fertilizer,1) \
                 * IRRIGATION_FX.get(irrigation,1) * (1+(float(ndvi)-0.6)*0.2)
    final = max(s['min'], min(s['max'], final))
    return {'yield_kg_ha': round(final), 'yield_quintal_ha': round(final/100,1),
            'range_min': s['min'], 'range_max': s['max'], 'range_mean': s['mean'],
            'unit': s.get('unit','kg/ha')}


def get_recommendations(crop, N, P, K, ph, temp, rain, moisture, disease, fertilizer, irrigation):
    info = CROP_INFO.get(crop, {})
    recs = []
    npk  = NPK_OPT.get(crop, {'N':(40,100),'P':(30,70),'K':(30,70)})
    if N < npk['N'][0]:
        recs.append(f"⚠️ Nitrogen deficient ({N:.0f} kg/ha) — apply Urea to reach {npk['N'][0]}–{npk['N'][1]} kg/ha")
    elif N > npk['N'][1]+20:
        recs.append(f"⚠️ Excess Nitrogen ({N:.0f} kg/ha) — reduce to avoid lodging/leaching")
    if P < npk['P'][0]:
        recs.append(f"⚠️ Phosphorus low — apply DAP/SSP to reach {npk['P'][0]}–{npk['P'][1]} kg/ha")
    if K < npk['K'][0]:
        recs.append(f"⚠️ Potassium low — apply MOP to reach {npk['K'][0]}–{npk['K'][1]} kg/ha")
    try:
        plo, phi = [float(x) for x in info.get('ph_range','6.0–7.5').split('–')]
        if ph < plo:   recs.append(f"⚠️ Soil pH {ph:.1f} is acidic — apply lime @ 200–300 kg/ha")
        elif ph > phi: recs.append(f"⚠️ Soil pH {ph:.1f} is alkaline — apply gypsum")
        else:          recs.append(f"✅ Soil pH {ph:.1f} is optimal for {crop}")
    except: pass
    if   disease == 'Severe':   recs.append("🚨 Severe disease — ~26% yield loss. Apply treatment immediately")
    elif disease == 'Moderate': recs.append("⚠️ Moderate disease — ~14% reduction. Treat within 1 week")
    elif disease == 'Mild':     recs.append("ℹ️ Mild disease — monitor closely, ~8% yield impact")
    if irrigation == 'None' and rain < 400:
        recs.append(f"⚠️ Low rainfall ({rain:.0f} mm) with no irrigation — consider drip")
    elif irrigation == 'Drip':
        recs.append("✅ Drip irrigation — saves 30–40% water vs flood irrigation")
    for t in info.get('tips',[])[:2]: recs.append(f"💡 {t}")
    for ft in info.get('fertilizer_tips',[])[:2]: recs.append(f"🌱 Recommended: {ft}")
    return recs[:8]
