def statisticalParity(class_index, sample, label_predict):
    num_c1 = sum(sample[:, class_index])
    num_c2 = len(label_predict) - num_c1
    
    num_c1_predict = len([e for e in range(0, len(label_predict)) if label_predict[e] == 1 and sample[e, class_index] == 1])
    num_c2_predict = len([e for e in range(0, len(label_predict)) if label_predict[e] == 1 and sample[e, class_index] == 0])
    
    return abs(num_c1_predict/num_c1 - num_c2_predict/num_c2)

def normedDisparate(class_index, sample, label_predict):
    num_c1 = sum(sample[:, class_index])
    num_c2 = len(label_predict) - num_c1
    
    num_c1_predict = len([e for e in range(0, len(label_predict)) if label_predict[e] == 1 and sample[e, class_index] == 1])
    num_c2_predict = len([e for e in range(0, len(label_predict)) if label_predict[e] == 1 and sample[e, class_index] == 0])
    
    prob_c1 = num_c1_predict/num_c1
    prob_c2 = num_c2_predict/num_c2
    
    if prob_c1 == 0 and prob_c2 == 0:
        return 0
    
    if prob_c1 < prob_c2:
        return 1 - (prob_c1 / prob_c2)
    else:
        return 1 - (prob_c2 / prob_c1)
