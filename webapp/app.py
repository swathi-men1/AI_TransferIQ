"""
TransferIQ Web App — Flask Backend
Uses best_model_v2.pkl (Stacking Ensemble, R²=0.761)
Returns single clean prediction + detailed player insights.
"""
from flask import Flask, jsonify, request, render_template
import pickle, numpy as np, pandas as pd
import os, math

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates'))

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', 'best_model_v2.pkl')
DATA_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'featured_dataset_final.csv')

print("Loading best_model_v2.pkl...")
with open(MODEL_PATH, 'rb') as f:
    BUNDLE = pickle.load(f)

MODELS   = BUNDLE['models']
META     = BUNDLE['meta_learner']
SCALER   = BUNDLE['scaler']
FEATURES = BUNDLE['features']

print("Loading player data...")
df_raw = pd.read_csv(DATA_PATH)
for col in ['Attacking','Skill','Movement']:
    df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce').fillna(0)

# Rebuild features
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
df_raw['log_hist_max']  = np.log1p(df_raw.get('max_market_val',  df_raw['last_market_val']))
df_raw['log_hist_mean'] = np.log1p(df_raw.get('mean_market_val', df_raw['last_market_val']))
df_raw['hist_growth']   = df_raw['avg_growth_rate'].clip(-1, 5)
df_raw['hist_seasons']  = df_raw['num_seasons']

for col in FEATURES:
    df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')
df_clean = df_raw.dropna(subset=FEATURES).copy()
MEDIANS  = df_clean[FEATURES].median().to_dict()

# Pre-compute predictions
X_all    = SCALER.transform(df_clean[FEATURES].values)
meta_all = np.column_stack([MODELS['gb'].predict(X_all), MODELS['et'].predict(X_all),
                             MODELS['rf'].predict(X_all), MODELS['ridge'].predict(X_all)])
df_clean['predicted_value'] = np.expm1(META.predict(meta_all))

# Dataset-level stats for percentile calculation
ALL_PREDS = df_clean['predicted_value'].values
ALL_VALS  = df_clean['last_market_val'].fillna(0).values

DISPLAY_COLS = ['Name','Age','OVA','Club','Nationality','BP','position_group',
                'career_stage','avg_sentiment','sentiment_label','performance_score',
                'potential_score','injury_risk_category','Contract_Years',
                'predicted_value','last_market_val','availability_score',
                'total_days_missed','total_injuries','avg_growth_rate','value_volatility',
                'physical_index','technical_index','is_peak_age']

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
        'availability':     float(round(row.get('availability_score', 1), 3)),
        'days_missed':      int(row.get('total_days_missed', 0)),
        'total_injuries':   int(row.get('total_injuries', 0)),
        'growth_rate':      float(round(row.get('avg_growth_rate', 0), 4)),
        'is_peak_age':      int(row.get('is_peak_age', 0)),
    }

ALL_PLAYERS = [build_player(row) for _, row in df_clean[DISPLAY_COLS].iterrows()]
print(f"Model v2 ready. {len(ALL_PLAYERS)} players. R²={BUNDLE['r2_test']:.4f}\n")


def derive_features(d):
    age  = float(d.get('age', 26))
    ova  = float(d.get('ova', 72))
    perf = float(d.get('performance_score', 48))
    pot  = float(d.get('potential_score', 74))
    pos  = str(d.get('position', 'midfielder'))
    inj  = str(d.get('injury_risk', 'low'))
    sent = float(d.get('sentiment', 0.0))
    cont = float(d.get('contract_years', 4))

    inj_map = {'none':(0,0,0,1.0),'low':(3,40,13.3,0.991),
               'medium':(8,120,15.0,0.973),'high':(25,400,16.0,0.910)}
    ti, td, adpi, avail = inj_map.get(inj, inj_map['low'])
    scale = perf / 48.0

    feat_vec = {
        'Age':age,'OVA':ova,'age_peak_diff':age-26,
        'is_peak_age':1.0 if 23<=age<=28 else 0.0,'age_squared':age**2,
        'performance_score':perf,'physical_index':67.5*scale,'technical_index':66.0*scale,
        'Attacking':min(437,302*scale),'Skill':min(433,316*scale),'Movement':min(450,345*scale),
        'Acceleration':min(353,69*scale),'Stamina':min(94,68*scale),'Strength':min(93,67*scale),
        'Short Passing':min(92,69*scale),'Shot Power':min(364,69*scale),
        'Heading Accuracy':min(92,60*scale),'Jumping':min(95,67*scale),
        'total_injuries':float(ti),'total_days_missed':float(td),'avg_days_per_injury':adpi,
        'injury_risk_score':adpi,'availability_score':avail,
        'avg_sentiment':sent,'sentiment_count':3.0,'sentiment_performance_index':perf*(1+sent*0.1),
        'Contract_Years':cont,'contract_value_proxy':cont*ova,
        'Height_cm':MEDIANS['Height_cm'],'Weight_kg':MEDIANS['Weight_kg'],'is_left_footed':0.0,
        'pos_defender':1.0 if pos=='defender' else 0.0,'pos_forward':1.0 if pos=='forward' else 0.0,
        'pos_goalkeeper':1.0 if pos=='goalkeeper' else 0.0,'pos_midfielder':1.0 if pos=='midfielder' else 0.0,
        'num_seasons':MEDIANS['num_seasons'],'avg_growth_rate':MEDIANS['avg_growth_rate'],
        'value_volatility':MEDIANS['value_volatility'],
        'OVA_norm':np.clip((ova-57)/(92-57),0,1),'performance_score_norm':np.clip((perf-30.7)/(62.4-30.7),0,1),
        'Age_norm':np.clip((age-16)/(39-16),0,1),'avg_sentiment_norm':np.clip((sent+0.7)/1.6,0,1),
        'injury_adj_performance':perf*avail,'potential_score':pot,
        'ova_x_potential':ova*pot,'ova_x_peak':ova*(1.0 if 23<=age<=28 else 0.0),
        'ova_squared':ova**2,'youth_flag':(1.0 if age<23 else 0.0)*ova,
        'perf_x_avail':perf*avail,'sent_x_ova':sent*ova,'contract_x_ova':cont*ova,
        'contract_x_potential':cont*pot,'inj_x_age':float(td)/(age+1),
        'perf_x_pot':perf*pot,'ova_x_contract':ova*np.log1p(cont),
        'log_hist_max':MEDIANS.get('log_hist_max',14.0),'log_hist_mean':MEDIANS.get('log_hist_mean',13.5),
        'hist_growth':0.05,'hist_seasons':float(MEDIANS.get('hist_seasons',3)),
    }
    return [feat_vec[f] for f in FEATURES]


def generate_insights(d, predicted_value):
    """Generate detailed player insights from inputs."""
    age   = float(d.get('age', 26))
    ova   = float(d.get('ova', 72))
    perf  = float(d.get('performance_score', 48))
    pot   = float(d.get('potential_score', 74))
    pos   = str(d.get('position', 'midfielder'))
    inj   = str(d.get('injury_risk', 'low'))
    sent  = float(d.get('sentiment', 0.0))
    cont  = float(d.get('contract_years', 4))

    inj_days = {'none':0,'low':40,'medium':120,'high':400}.get(inj,40)
    avail = {'none':1.0,'low':0.991,'medium':0.973,'high':0.910}.get(inj,0.991)

    # Percentile in dataset
    pct = float(np.mean(ALL_PREDS < predicted_value) * 100)

    # Value drivers — what's boosting/hurting value
    drivers = []

    # Age insight
    if 23 <= age <= 28:
        drivers.append({'factor':'Age','impact':'positive','score':95,
            'detail':f'Age {int(age)} — peak transfer age zone (23-28). Buyers pay maximum premium.'})
    elif age < 23:
        drivers.append({'factor':'Age','impact':'positive','score':80,
            'detail':f'Age {int(age)} — young talent. High potential premium adds to value.'})
    elif age <= 32:
        drivers.append({'factor':'Age','impact':'neutral','score':50,
            'detail':f'Age {int(age)} — experienced player. Value starts declining from peak.'})
    else:
        drivers.append({'factor':'Age','impact':'negative','score':20,
            'detail':f'Age {int(age)} — veteran stage. Transfer value significantly reduced.'})

    # OVA insight
    ova_pct = (ova - 57) / (92 - 57) * 100
    drivers.append({'factor':'OVA Rating','impact':'positive' if ova>=75 else 'neutral' if ova>=68 else 'negative',
        'score':int(ova_pct),
        'detail':f'OVA {int(ova)}/100 — {"elite" if ova>=85 else "very good" if ova>=78 else "good" if ova>=72 else "average"} rating. {"Top 15% of players." if ova>=82 else "Above average." if ova>=75 else "Room to grow."}'})

    # Performance insight
    perf_pct = (perf - 30.7) / (62.4 - 30.7) * 100
    drivers.append({'factor':'Performance Score','impact':'positive' if perf>=55 else 'neutral' if perf>=45 else 'negative',
        'score':int(perf_pct),
        'detail':f'Score {perf:.1f}/62 — composite of attacking, skill, movement stats. {"Excellent all-round performer." if perf>=55 else "Solid performer." if perf>=48 else "Below average output."}'})

    # Potential insight
    pot_diff = pot - ova
    drivers.append({'factor':'Potential Score','impact':'positive' if pot>=80 else 'neutral' if pot>=72 else 'negative',
        'score':int((pot-57)/(92-57)*100),
        'detail':f'Potential {int(pot)}/100. {"Significant growth expected!" if pot_diff>5 else "Already at potential." if pot_diff<=0 else "Some growth possible."} {"High future sell-on value." if pot>=82 and age<26 else ""}'})

    # Injury insight
    inj_colors = {'none':'positive','low':'positive','medium':'neutral','high':'negative'}
    inj_scores = {'none':100,'low':85,'medium':55,'high':20}
    inj_details = {
        'none': 'No injury history. Full availability — clubs pay premium for reliability.',
        'low':  f'Minor injuries only (~{inj_days} days missed). Good availability ({avail*100:.0f}%).',
        'medium': f'Regular injuries (~{inj_days} days missed). Availability {avail*100:.0f}% — moderate risk discount applied.',
        'high': f'Frequent/serious injuries (~{inj_days} days missed). Only {avail*100:.0f}% available — significant value reduction.'
    }
    drivers.append({'factor':'Injury Risk','impact':inj_colors[inj],'score':inj_scores[inj],
        'detail':inj_details[inj]})

    # Sentiment insight
    sent_score = int((sent + 1) / 2 * 100)
    if sent > 0.2:
        sent_detail = f'Score +{sent:.2f} — strong positive media buzz. Fan favourite status adds ~{int(sent*15)}% value premium.'
        sent_impact = 'positive'
    elif sent > 0.05:
        sent_detail = f'Score +{sent:.2f} — positive public perception. Minor value boost applied.'
        sent_impact = 'positive'
    elif sent < -0.2:
        sent_detail = f'Score {sent:.2f} — negative media coverage. Controversy reduces transfer appeal by ~{int(abs(sent)*15)}%.'
        sent_impact = 'negative'
    elif sent < -0.05:
        sent_detail = f'Score {sent:.2f} — slightly negative sentiment. Small value reduction.'
        sent_impact = 'negative'
    else:
        sent_detail = 'Neutral media presence. No significant sentiment premium or discount.'
        sent_impact = 'neutral'
    drivers.append({'factor':'Media Sentiment','impact':sent_impact,'score':sent_score,'detail':sent_detail})

    # Contract insight
    if cont >= 3:
        cont_impact = 'positive'
        cont_score  = min(100, int(cont/8*100))
        cont_detail = f'{cont:.0f} years remaining — long contract = buying club must pay higher fee. Strong leverage for seller.'
    elif cont >= 1:
        cont_impact = 'neutral'
        cont_score  = int(cont/8*100)
        cont_detail = f'{cont:.0f} year(s) remaining — moderate contract length. Standard negotiation position.'
    else:
        cont_impact = 'negative'
        cont_score  = 5
        cont_detail = 'Short or expired contract — player can leave on a free soon. Transfer value near zero.'
    drivers.append({'factor':'Contract Length','impact':cont_impact,'score':cont_score,'detail':cont_detail})

    # Position insight
    pos_vals = {'forward':3.8,'midfielder':3.2,'defender':2.1,'goalkeeper':1.8}
    pos_label = {'forward':'Forwards','midfielder':'Midfielders','defender':'Defenders','goalkeeper':'Goalkeepers'}
    pos_avg = pos_vals.get(pos, 3.0)
    drivers.append({'factor':'Position','impact':'positive' if pos in ['forward','midfielder'] else 'neutral',
        'score':int(pos_vals.get(pos,2.5)/4.5*100),
        'detail':f'{pos_label.get(pos,pos).rstrip("s")} position. {pos_label.get(pos,"Players")} avg market value ~€{pos_avg}M. {"High-demand position." if pos in ["forward","midfielder"] else "Specialist position."}'})

    # Overall value rating
    positive = sum(1 for d in drivers if d['impact']=='positive')
    negative = sum(1 for d in drivers if d['impact']=='negative')
    overall_score = int(np.mean([d['score'] for d in drivers]))

    if overall_score >= 75:
        rating = 'Elite Transfer Target'
        rating_color = 'success'
    elif overall_score >= 55:
        rating = 'Strong Transfer Candidate'
        rating_color = 'info'
    elif overall_score >= 35:
        rating = 'Average Market Value'
        rating_color = 'warning'
    else:
        rating = 'Below Market Average'
        rating_color = 'danger'

    return {
        'overall_score':  overall_score,
        'overall_rating': rating,
        'rating_color':   rating_color,
        'percentile':     round(pct, 1),
        'positive_factors': positive,
        'negative_factors': negative,
        'drivers':        drivers,
        'summary': f"This {pos} aged {int(age)} with OVA {int(ova)} is ranked in the top {100-int(pct):.0f}% of transfer targets. "
                   f"{'Peak-age premium applies. ' if 23<=age<=28 else ''}"
                   f"{'Injury history discounts value. ' if inj in ['medium','high'] else ''}"
                   f"{'Positive sentiment adds market appeal.' if sent>0.05 else ''}"
    }


# ── Routes ──────────────────────────────────────────────────────────

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
        fv  = derive_features(data)
        X   = SCALER.transform([fv])

        # Sirf Ridge Regression ka prediction
        ridge_log = float(MODELS['ridge'].predict(X)[0])
        pred_val  = int(np.expm1(ridge_log))

        # Stats for visualization
        age   = float(data.get('age', 26))
        ova   = float(data.get('ova', 72))
        perf  = float(data.get('performance_score', 48))
        pot   = float(data.get('potential_score', 74))
        inj   = str(data.get('injury_risk', 'low'))
        sent  = float(data.get('sentiment', 0.0))
        cont  = float(data.get('contract_years', 4))
        avail = {'none':1.0,'low':0.991,'medium':0.973,'high':0.910}.get(inj, 0.991)

        # Percentile in dataset
        pct = float(np.mean(ALL_PREDS < pred_val) * 100)

        # Normalized scores (0-100) for radar/bar chart
        stats = {
            'OVA Rating':        int((ova - 57) / (92 - 57) * 100),
            'Performance':       int((perf - 30.7) / (62.4 - 30.7) * 100),
            'Potential':         int((pot - 57) / (92 - 57) * 100),
            'Availability':      int(avail * 100),
            'Sentiment':         int((sent + 1) / 2 * 100),
            'Contract Value':    int(min(cont / 10, 1) * 100),
        }

        # Age insight
        if 23 <= age <= 28:
            age_label = f'Peak age ({int(age)}) — max market premium'
            age_color = 'positive'
        elif age < 23:
            age_label = f'Young talent ({int(age)}) — future potential valued'
            age_color = 'positive'
        elif age <= 32:
            age_label = f'Experienced ({int(age)}) — value declining from peak'
            age_color = 'neutral'
        else:
            age_label = f'Veteran ({int(age)}) — significant value reduction'
            age_color = 'negative'

        inj_label = {
            'none':   'No injury history — clubs pay full premium',
            'low':    'Low risk — minor injuries, 99% availability',
            'medium': 'Medium risk — regular injuries reduce value',
            'high':   'High risk — serious injuries, ~9% value discount',
        }.get(inj, '')

        if sent > 0.05:
            sent_label = f'Positive (+{sent:.2f}) — fan favourite adds market appeal'
        elif sent < -0.05:
            sent_label = f'Negative ({sent:.2f}) — controversy reduces transfer demand'
        else:
            sent_label = 'Neutral — no significant media premium'

        return jsonify({
            'predicted_value': pred_val,
            'model':           'Ridge Regression',
            'percentile':      round(pct, 1),
            'stats':           stats,
            'labels': {
                'age':     age_label,
                'age_color': age_color,
                'injury':  inj_label,
                'sentiment': sent_label,
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats')
def get_stats():
    vals = [p['predicted_value'] for p in ALL_PLAYERS]
    return jsonify({
        'total_players': len(ALL_PLAYERS),
        'avg_predicted': int(np.mean(vals)),
        'max_predicted': int(np.max(vals)),
        'model_r2':      round(BUNDLE['r2_test'], 4),
        'model_name':    'Stacking Ensemble v2',
        'features_used': len(FEATURES),
        'accuracy_w10':  '60.9%',
        'accuracy_w25':  '77.8%',
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)
