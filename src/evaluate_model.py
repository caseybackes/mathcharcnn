import numpy as np 

def evaluate_model(X_test,y_test,categories,model, limit=-1,return_prediction_array=True):
    predictions_cat_ix = [np.argmax(row) for row in model.predict(X_test[0:limit])]
    predictions_class_names = [categories[x] for x in predictions_cat_ix]
    y_true_ix = [categories[x] for x in [np.argmax(yi) for yi in y_test[0:limit]]]
    result = np.array([pred==truth for pred,truth in list(zip(predictions_class_names,y_true_ix))])
    acc = result.sum()/len(result)
    
    if return_prediction_array:
        return acc, result
    else:
        return acc

