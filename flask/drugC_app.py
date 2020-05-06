import flask
import pickle
import pandas as pd
from xgboost import XGBClassifier


verbose=False

#%%---------- Load Models ----------------#
classifiers = ['alcohol', 'choc', 'caff',
    'nicotine', 'cannabis', 'lsd', 'mushrooms',
    'legalh', 'ketamine', 'ecstasy', 'amphet',
    'coke', 'crack', 'heroin', 'meth',
    'benzos', 'amyl', 'vsa', 'monthly_illicit_user']
final_models = {}
path = './data/'
for classifier in classifiers:
    with open(path+classifier+'_xgb.pkl', 'rb') as handle:
        final_models[classifier]= pickle.load(handle)
    handle.close()




#%% Initialize the app
app = flask.Flask(__name__)

# Homepage
@app.route("/")
def viz_page():
    """
    Homepage: serve our visualization page, awesome.html
    """
    with open("awesome.html", 'r') as viz_file:
        return viz_file.read()

def age_transform(age):
    if   age in range(18,25): return -0.95197
    elif age in range(25,35): return -0.07854
    elif age in range(35,45): return 0.49788
    elif age in range(45,55): return 1.09449
    elif age in range(55,65): return 1.82213
    elif age in range(65,101): return 2.59171
    else: return 2.59171

def education_transform(edu_lvl:int):
    edu_lvls = {0: -2.43591, 1: -1.73790, 2: -1.43719,
    3: -1.22751, 4: -0.61113, 5: -0.05921, 
    6: 0.45468, 7: 1.16365, 8: 1.98437}
    return edu_lvls[edu_lvl]

def nscore_transform(nscore:int):
    nscores = {12: -3.46436, 13: -3.15735, 14: -2.75696,
    15: -2.52197, 16: -2.42317, 17: -2.34360, 18: -2.21844,
    19: -2.05048, 20: -1.86962, 21: -1.69163, 22: -1.55078,
    23: -1.43907, 24: -1.32828, 25: -1.19430, 26: -1.05308,
    27: -0.92104, 28: -0.79151, 29: -0.67825, 30: -0.58016,
    31: -0.46725, 32: -0.34799, 33: -0.24649, 34: -0.14882,
    35: -0.05188, 36: 0.04257, 37: 0.13606, 38: 0.22393,
    39: 0.31287, 40: 0.41667, 41: 0.52135, 42: 0.62967,
    43: 0.73545, 44: 0.82563, 45: 0.91093, 46: 1.02119,
    47: 1.13281, 48: 1.23461, 49: 1.37297, 50: 1.49158,
    51: 1.60383, 52: 1.72012, 53: 1.83990, 54: 1.98437,
    55: 2.12700, 56: 2.28554, 57: 2.46262, 58: 2.61139,
    59: 2.82196, 60: 3.27393}
    return nscores[nscore]

def escore_transform(escore:int):
    escores = {16: -3.27393, 17: -3.27393, 18: -3.00537,
    19: -2.72827, 20: -2.53830, 21: -2.44904, 22: -2.32338,
    23: -2.21069, 24: -2.11437, 25: -2.03972, 26: -1.92173,
    27: -1.76250, 28: -1.63340, 29: -1.50796, 30: -1.37639,
    31: -1.23177, 32: -1.09207, 33: -0.94779, 34: -0.80615,
    35: -0.69509, 36: -0.57545, 37: -0.43999, 38: -0.30033,
    39: -0.15487, 40: 0.00332, 41: 0.16767, 42: 0.32197,
    43: 0.47617, 44: 0.63779, 45: 0.80523, 46: 0.96248,
    47: 1.11406, 48: 1.28610, 49: 1.45421, 50: 1.58487,
    51: 1.74091, 52: 1.93886, 53: 2.12700, 54: 2.32338,
    55: 2.57309, 56: 2.85950, 57: 2.85950, 58: 3.00537,
    59: 3.27393, 60: 3.27393}
    return escores[escore]

def oscore_transform(oscore:int):
    oscores = {24: -3.27393, 25: -2.85950, 26: -2.85950,
    27: -2.63199, 28: -2.63199, 29: -2.39883, 30: -2.21069,
    31: -2.09015, 32: -1.97495, 33: -1.82919, 34: -1.68062,
    35: -1.55521, 36: -1.42424, 37: -1.27553, 38: -1.11902,
    39: -0.97631, 40: -0.84732, 41: -0.71727, 42: -0.58331,
    43: -0.45174, 44: -0.31776, 45: -0.17779, 46: -0.01928,
    47: 0.14143, 48: 0.29338, 49: 0.44585, 50: 0.58331,
    51: 0.72330, 52: 0.88309, 53: 1.06238, 54: 1.24033,
    55: 1.43533, 56: 1.65653, 57: 1.88511, 58: 2.15324,
    59: 2.44904, 60: 2.90161}
    return oscores[oscore]

def ascore_transform(ascore:int):
    ascores = {12: -3.46436, 13: -3.46436, 14: -3.15735,
    15: -3.15735, 16:-3.15735, 17: -3.00537, 18: -3.00537,
    19: -3.00537, 20: -3.00537, 21: -2.90161, 22: -2.90161,
    23: -2.90161, 24: -2.78793, 25: -2.70172, 26: -2.53830,
    27: -2.35413, 28: -2.21844, 29: -2.07848, 30: -1.92595,
    31: -1.77200, 32: -1.62090, 33: -1.47955, 34: -1.34289,
    35: -1.21213, 36: -1.07533, 37: -0.91699, 38: -0.76096,
    39: -0.60633, 40: -0.45321, 41: -0.30172, 42: -0.15487,
    43: -0.01729, 44: 0.13136, 45: 0.28783, 46: 0.43852,
    47: 0.59042, 48: 0.76096, 49: 0.94156, 50: 1.11406,
    51: 1.2861, 52: 1.45039, 53: 1.61108, 54: 1.81866,
    55: 2.03972, 56: 2.23427, 57: 2.46262, 58: 2.75696,
    59: 3.15735, 60: 3.46436}
    return ascores[ascore]

def cscore_transform(cscore:int):
    cscores = {17: -3.46436, 18: -3.15735, 19: -3.15735,
    20: -2.90161, 21: -2.72827, 22: -2.57309, 23: -2.42317,
    24: -2.30408, 25: -2.18109, 26: -2.04506, 27: -1.92173,
    28: -1.78169, 29: -1.64101, 30: -1.51840, 31: -1.38502,
    32: -1.25773, 33: -1.13788, 34: -1.01450, 35: -0.89891,
    36: -0.78155, 37: -0.65253, 38: -0.52745, 39: -0.40581,
    40: -0.27607, 41: -0.14277, 42: -0.00665, 43: 0.12331,
    44: 0.25953, 45: 0.41594, 46: 0.58489, 47: 0.7583,
    48: 0.93949, 49: 1.13407, 50: 1.30612, 51: 1.46191,
    52: 1.63088, 53: 1.81175, 54: 2.04506, 55: 2.33337,
    56: 2.63199, 57: 3.00537, 58: 3.46436}
    return cscores[cscore]

def imp_transform(imp:int):
    imps = {-4: -2.55524, -3: -1.37983, -2: -0.71126,
    -1: -0.21712, 0: 0.19268, 1: 0.52975, 2: 0.88113,
    3: 1.29221, 4: 1.86203, 5: 2.90161}
    return imps[imp]

def ss_transform(ss):
    ss_s = {-5: -2.07848, -4: -1.54858, -3: -1.18084,
    -2: -0.84637, -1: -0.52593, 0: -0.21575, 1: 0.07987,
    2: 0.40148, 3: 0.76540, 4: 1.22470, 5: 1.92173}
    return ss_s[ss]

def transform_inputs(inputs: dict):
    """
    Takes in input list with values corresponding to
    age, education, nscore, escore, oscore, ascore,
    cscore, impulsiveness, and ss (in that order).

    Returns a pandas dataframe of transformed values
    """
    inputs['age'] = age_transform(int(inputs['age']))
    inputs['education'] = education_transform(int(inputs['education']))
    inputs['nscore'] = nscore_transform(int(inputs['nscore']))
    inputs['escore'] = escore_transform(int(inputs['escore']))
    inputs['oscore'] = oscore_transform(int(inputs['oscore']))
    inputs['ascore'] = ascore_transform(int(inputs['ascore']))
    inputs['cscore'] = cscore_transform(int(inputs['cscore']))
    inputs['impulsiveness'] = imp_transform(int(inputs['impulsiveness']))
    inputs['ss'] = ss_transform(int(inputs['ss']))

    return inputs


def choose_binary(value, threshold):
    if value <= threshold:
        return 0
    else:
        return 1

# Get an example and return its score from the predictor model
@app.route("/score", methods=["POST"])
def score():
    """
    When A POST request with json data is made to this url,
    Read the example from the json, predict probability and
    send it with a response
    """
    results = {'alcohol': None, 'choc': None, 'caff': None,
    'nicotine': None, 'cannabis': None, 'lsd': None, 'mushrooms': None,
    'legalh': None, 'ketamine': None, 'ecstasy': None, 'amphet': None,
    'coke': None, 'crack': None, 'heroin': None, 'meth': None,
    'benzos': None, 'amyl': None, 'vsa': None,}

    # Get decision score for our example that came with the request
    
    data = flask.request.get_json()
    
    if verbose:
        print("+++++++++++++++++++++++++++++++++++")
        print("+++++++++++++++++++++++++++++++++++")
        print(flask.request.is_json)
        print(data)
        print("+++++++++++++++++++++++++++++++++++")

    transformed_data = transform_inputs(data)

    if verbose:
        print("+++++++++++++++++++++++++++++++++++")
        print(transformed_data)
        print("+++++++++++++++++++++++++++++++++++")
        print("+++++++++++++++++++++++++++++++++++")

    # stuff transformed data into dataframe of features
    features_lvl1 = pd.DataFrame([transformed_data])

    # Level 1
    for drug in ['alcohol','caff','choc','monthly_illicit_user']:
        data_lvl1 = final_models[drug].predict_proba(features_lvl1)
        results[drug] = data_lvl1[0,1]

    # Level 2
    # Create new features df, delete old to save memory
    features_lvl2 = features_lvl1
    del features_lvl1
    # Define lvl2 drugs
    lvl2_drugs = ['amphet', 'benzos', 'cannabis', 'ecstasy', 'nicotine']
    # use results of lvl1 'monthly_illicit_user' to predict a class
    monthly_user_threshold = 0.5
    features_lvl2['monthly_illicit_user'] = choose_binary(
        results['monthly_illicit_user'], monthly_user_threshold)
    # Predict lvl2 drugs
    for drug in lvl2_drugs:
        data_lvl2 = final_models[drug].predict_proba(features_lvl2)
        results[drug] = data_lvl2[0,1]

    # Level 3
    # Create new features df, delete old to save memory
    features_lvl3 = features_lvl2
    del features_lvl2
    # Define lvl3 drugs
    lvl3_drugs = ['coke', 'meth']
    # Append results from lvl2 drugs to features_lvl3
    threshold=0.5
    for drug in lvl2_drugs:
        features_lvl3[drug] = choose_binary(
            results[drug], threshold)
    # Predict lvl3 drugs
    for drug in lvl3_drugs:
        data_lvl3 = final_models[drug].predict_proba(features_lvl3)
        results[drug]=data_lvl3[0, 1]

    # Level 3
    # Create new features df, delete old to save memory
    features_lvl4 = features_lvl3
    del features_lvl3
    # Define lvl3 drugs
    lvl4_drugs = ['amyl','heroin','crack','ketamine','legalh','mushrooms','lsd','vsa']
    # Append results from lvl2 drugs to features_lvl3
    threshold=0.5
    for drug in lvl3_drugs:
        features_lvl4[drug] = choose_binary(
                                results[drug], threshold)
    # Predict lvl4 drugs
    for drug in lvl4_drugs:
        data_lvl4=final_models[drug].predict_proba(features_lvl4)
        results[drug]=data_lvl4[0, 1]

    # Convert floats to strings, or else jsonify throws a tantrum
    # because it doesn't recognize numpy's float32
    del results['monthly_illicit_user']
    for drug in results.keys():
        results[drug] = str(results[drug])

    return flask.jsonify(results)


#%%--------- RUN WEB APP SERVER ------------#
# Start the app server on port XXXX
# (The default website port)
if __name__ == '__main__':
    app.run(debug=True)
