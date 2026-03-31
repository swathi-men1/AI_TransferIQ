"""
TransferIQ Web App — Flask Backend
Loads best_model_v2.pkl (R²=0.761) and serves predictions via REST API.
"""
from flask import Flask, jsonify, request, render_template
import pickle, numpy as np, pandas as pd
import os, math

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'best_model_v2.pkl')
DATA_PATH  = os.path.join(os.path.dirname(__file__), 'data',   'featured_dataset_final.csv')

print("Loading best_model_v2.pkl...")
with open(MODEL_PATH, 'rb') as f:
    BUNDLE = pickle.load(f)

MODELS   = BUNDLE['models']
META     = BUNDLE['meta_learner']
SCALER   = BUNDLE['scaler']
FEATURES = BUNDLE['features']

print("Loading player data...")
df_raw = pd.read_csv(DATA_PATH)

# Rebuild interaction + hist features (same as training)
for col in ['Attacking','Skill','Movement']:
    df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce').fillna(0)
df_raw['ova_x_potential']      = df_raw['OVA'] * df_raw['potential_score']
df_raw['ova_x_peak']           = df_raw['OVA'] * df_raw['is_peak_age']
df_raw['ova_squared']          = df_raw['OVA'] ** 2
df_raw['youth_flag']           = (df_raw['Age'] < 23).astype(int) * df_raw['OVA']
df_raw['perf_x_avail']         = df_raw['performance_score'] * df_raw['availability_score']
df_raw['sent_x_ova']           = df_raw['avg_sentiment'] * df_raw['OVA']
df_raw['contract_x_ova']       = df_raw['Contract_Years'] * df_raw['OVA']
df_raw['contract_x_potential'] = df_raw['Contract_Years'] * df_raw['potential_score']
df_raw['inj_x_age']            = df_raw['total_days_missed'] / (df_raw['Age'] + 1)
df_raw['perf_x_pot']           = df_raw['performance_score'] * df_raw['potential_score']
df_raw['ova_x_contract']       = df_raw['OVA'] * np.log1p(df_raw['Contract_Years'])

# Hist market features — use avg_growth_rate as proxy
df_raw['log_hist_max']  = np.log1p(df_raw['max_market_val'])  if 'max_market_val'  in df_raw.columns else np.log1p(df_raw['last_market_val'])
df_raw['log_hist_mean'] = np.log1p(df_raw['mean_market_val']) if 'mean_market_val' in df_raw.columns else np.log1p(df_raw['last_market_val'])
df_raw['hist_growth']   = df_raw['avg_growth_rate'].clip(-1, 5)
df_raw['hist_seasons']  = df_raw['num_seasons']

for col in FEATURES:
    df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')
df_clean = df_raw.dropna(subset=FEATURES).copy()
MEDIANS  = df_clean[FEATURES].median().to_dict()

# Pre-compute all player predictions
X_all = SCALER.transform(df_clean[FEATURES].values)
meta_all = np.column_stack([MODELS['gb'].predict(X_all), MODELS['et'].predict(X_all),
                             MODELS['rf'].predict(X_all), MODELS['ridge'].predict(X_all)])
df_clean['predicted_value'] = np.expm1(META.predict(meta_all))

DISPLAY_COLS = ['Name','Age','OVA','Club','Nationality','BP','position_group',
                'career_stage','avg_sentiment','sentiment_label','performance_score',
                'potential_score','injury_risk_category','Contract_Years',
                'predicted_value','last_market_val']

def build_player(row):
    return {
        'name':             str(row.get('Name','')),
        'age':              int(row.get('Age', 0)),
        'ova':              int(row.get('OVA', 0)),
        'club':             str(row.get('Club','')),
        'nationality':      str(row.get('Nationality','')),
        'position':         str(row.get('BP','')),
        'position_group':   str(row.get('position_group','')),
        'career_stage':     str(row.get('career_stage','')),
        'sentiment':        float(round(row.get('avg_sentiment', 0), 3)),
        'sentiment_label':  str(row.get('sentiment_label','neutral')),
        'performance_score':float(round(row.get('performance_score', 0), 1)),
        'potential_score':  float(round(row.get('potential_score', 0), 1)),
        'injury_risk':      str(row.get('injury_risk_category','none')),
        'contract_years':   float(row.get('Contract_Years', 0) or 0),
        'predicted_value':  int(row.get('predicted_value', 0)),
        'actual_value':     int(row.get('last_market_val', 0) or 0),
    }

ALL_PLAYERS = [build_player(row) for _, row in df_clean[DISPLAY_COLS].iterrows()]
print(f"Model v2 loaded. {len(ALL_PLAYERS)} players ready. R²={BUNDLE['r2_test']:.4f}\n")


def derive_features(d):
    age     = float(d.get('age', 26))
    ova     = float(d.get('ova', 72))
    perf    = float(d.get('performance_score', 48))
    pot     = float(d.get('potential_score', 74))
    pos     = str(d.get('position', 'midfielder'))
    inj     = str(d.get('injury_risk', 'low'))
    sent    = float(d.get('sentiment', 0.0))
    cont    = float(d.get('contract_years', 4))

    inj_map = {'none':(0,0,0,1.0),'low':(3,40,13.3,0.991),
               'medium':(8,120,15.0,0.973),'high':(25,400,16.0,0.910)}
    ti, td, adpi, avail = inj_map.get(inj, inj_map['low'])

    scale = perf / 48.0
    feat_vec = {
        'Age':age,'OVA':ova,'age_peak_diff':age-26,
        'is_peak_age':1.0 if 23<=age<=28 else 0.0,
        'age_squared':age**2,
        'performance_score':perf,'physical_index':67.5*scale,
        'technical_index':66.0*scale,
        'Attacking':min(437,302*scale),'Skill':min(433,316*scale),
        'Movement':min(450,345*scale),'Acceleration':min(353,69*scale),
        'Stamina':min(94,68*scale),'Strength':min(93,67*scale),
        'Short Passing':min(92,69*scale),'Shot Power':min(364,69*scale),
        'Heading Accuracy':min(92,60*scale),'Jumping':min(95,67*scale),
        'total_injuries':float(ti),'total_days_missed':float(td),
        'avg_days_per_injury':adpi,'injury_risk_score':adpi,'availability_score':avail,
        'avg_sentiment':sent,'sentiment_count':3.0,
        'sentiment_performance_index':perf*(1+sent*0.1),
        'Contract_Years':cont,'contract_value_proxy':cont*ova,
        'Height_cm':MEDIANS['Height_cm'],'Weight_kg':MEDIANS['Weight_kg'],
        'is_left_footed':0.0,
        'pos_defender':1.0 if pos=='defender' else 0.0,
        'pos_forward':1.0 if pos=='forward' else 0.0,
        'pos_goalkeeper':1.0 if pos=='goalkeeper' else 0.0,
        'pos_midfielder':1.0 if pos=='midfielder' else 0.0,
        'num_seasons':MEDIANS['num_seasons'],
        'avg_growth_rate':MEDIANS['avg_growth_rate'],
        'value_volatility':MEDIANS['value_volatility'],
        'OVA_norm':np.clip((ova-57)/(92-57),0,1),
        'performance_score_norm':np.clip((perf-30.7)/(62.4-30.7),0,1),
        'Age_norm':np.clip((age-16)/(39-16),0,1),
        'avg_sentiment_norm':np.clip((sent+0.7)/1.6,0,1),
        'injury_adj_performance':perf*avail,
        'potential_score':pot,
        # Interaction features
        'ova_x_potential':ova*pot,'ova_x_peak':ova*(1.0 if 23<=age<=28 else 0.0),
        'ova_squared':ova**2,'youth_flag':(1.0 if age<23 else 0.0)*ova,
        'perf_x_avail':perf*avail,'sent_x_ova':sent*ova,
        'contract_x_ova':cont*ova,'contract_x_potential':cont*pot,
        'inj_x_age':float(td)/(age+1),'perf_x_pot':perf*pot,
        'ova_x_contract':ova*np.log1p(cont),
        # Historical features (use medians as proxy for new players)
        'log_hist_max':MEDIANS.get('log_hist_max', np.log1p(ova*300000)),
        'log_hist_mean':MEDIANS.get('log_hist_mean', np.log1p(ova*200000)),
        'hist_growth':0.05,'hist_seasons':float(MEDIANS.get('hist_seasons',3)),
    }
    return [feat_vec[f] for f in FEATURES]


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/players')
def get_players():
    q     = request.args.get('q','').lower().strip()
    pos   = request.args.get('position','')
    stage = request.args.get('stage','')
    risk  = request.args.get('risk','')
    sent  = request.args.get('sentiment','')
    sort  = request.args.get('sort','predicted_value')
    page  = int(request.args.get('page',1))
    per   = int(request.args.get('per',24))

    res = ALL_PLAYERS
    if q:     res = [p for p in res if q in p['name'].lower() or q in p['club'].lower() or q in p['nationality'].lower()]
    if pos:   res = [p for p in res if p['position_group']==pos]
    if stage: res = [p for p in res if p['career_stage']==stage]
    if risk:  res = [p for p in res if p['injury_risk']==risk]
    if sent:  res = [p for p in res if p['sentiment_label']==sent]
    if sort in {'predicted_value','ova','age','performance_score','potential_score'}:
        res = sorted(res, key=lambda x: x.get(sort,0), reverse=True)

    total = len(res)
    start = (page-1)*per
    return jsonify({'players':res[start:start+per],'total':total,
                    'page':page,'per':per,'pages':math.ceil(total/per)})


@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({'error':'No data provided'}), 400
    try:
        fv = derive_features(data)
        X  = SCALER.transform([fv])
        meta = np.column_stack([MODELS['gb'].predict(X), MODELS['et'].predict(X),
                                 MODELS['rf'].predict(X), MODELS['ridge'].predict(X)])
        pred = float(np.expm1(META.predict(meta)[0]))
        age = float(data.get('age',26)); inj = str(data.get('injury_risk','low')); s = float(data.get('sentiment',0))
        return jsonify({
            'predicted_value': int(pred),
            'model_breakdown': {
                'gradient_boosting': {'value':int(np.expm1(MODELS['gb'].predict(X)[0])), 'weight':30},
                'extra_trees':       {'value':int(np.expm1(MODELS['et'].predict(X)[0])), 'weight':30},
                'random_forest':     {'value':int(np.expm1(MODELS['rf'].predict(X)[0])), 'weight':30},
                'ridge_regression':  {'value':int(np.expm1(MODELS['ridge'].predict(X)[0])), 'weight':10},
            },
            'insights': {
                'age_factor':      'Peak age (23-28) ✓' if 23<=age<=28 else 'Off-peak',
                'injury_impact':   {'none':'No history','low':'Minimal (-1%)','medium':'Moderate (-3%)','high':'Significant (-9%)'}.get(inj,''),
                'sentiment_impact': '+ve buzz boosts value' if s>0.05 else ('-ve sentiment reduces' if s<-0.05 else 'Neutral coverage'),
            },
            'model_version': 'v2',
            'model_r2':      BUNDLE['r2_test'],
        })
    except Exception as e:
        return jsonify({'error':str(e)}), 500


@app.route('/api/stats')
def get_stats():
    vals = [p['predicted_value'] for p in ALL_PLAYERS]
    return jsonify({
        'total_players': len(ALL_PLAYERS),
        'avg_predicted': int(np.mean(vals)),
        'max_predicted': int(np.max(vals)),
        'model_r2':      round(BUNDLE['r2_test'], 4),
        'model_name':    'Stacking Ensemble v2 (GB + ExtraTrees + RF + Ridge)',
        'features_used': len(FEATURES),
        'accuracy_w10':  '60.9%',
        'accuracy_w25':  '77.8%',
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)
