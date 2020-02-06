import os 

def class_count(list_of_class_strings, data_directory,verbose_only=False):
    class_dict = dict()
    for class_item in list_of_class_strings:
        path = os.path.join(data_directory, class_item) 
        class_dict[class_item] = len(os.listdir(path))
    if verbose_only:
        print(class_dict)
    else:
        return class_dict


