from sklearn.model_selection import train_test_split

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def dataset_generator(data_dict, size, cifar_type, seed=100):
    if cifar_type == 'cifar10':
        labels = b'labels'
    elif cifar_type == 'cifar100':
        labels = b'fine_labels'

    assert size <= len(data_dict[labels])
    ratio = float(size/len(data_dict[labels]))
    _, _X, _, _y = train_test_split(data_dict[b'data'],
                                    data_dict[labels],
                                    test_size = ratio,
                                    random_state = seed)

    dataset = {'data': _X, 'labels': _y}
    return dataset