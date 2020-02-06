from datetime import datetime as dt 
from contextlib import redirect_stdout

def save_model(model, categories, val_split, acc, hist):
    today = str(dt.now().date())
    timestamp = str(dt.now().date()) + "T:"+ str(dt.now().time())[0:8] 
    model.save(f'../models/simpleCNN-{timestamp}.h5')  # creates a HDF5 file 'my_model.h5'
    print(f"Saved as models/simpleCNN-{timestamp}.h5")
    with open(f'../models/reports/simpleCNN-{timestamp}.txt','w') as f:
        f.write(f'Classes in data: {categories}\n')
        f.write(f'Model: {str(model.name).capitalize()}')
        f.write(f'Train-to-Holdout Ratio: {1-val_split}\n')
        f.write(f'Holdout Accuracy: {acc}\n')
        f.write(f'Model.History:\n')
        #histkeys = hist.history.keys()
        for k,v in hist.history.items():
            f.write(f'{k}: {v}\n')
        with redirect_stdout(f):
                model.summary()
        f.write(f'Model.to_json\n{model.to_json()}')
        f.close()
        print(f"Report of model saved at models/reports/simpleCNN-{timestamp}.txt ")