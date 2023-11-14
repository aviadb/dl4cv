def predict(nnet, pre_proc_img, topk=3):
    global weights
    pred = nnet(pre_proc_img.unsqueeze(0)).squeeze(0).softmax(0)
    # class_id = pred.argmax().item()
    # cat_list = [ weights.meta["categories"][idx] for idx in topk_idx ]
    [topk_scores, topk_idx] = pred.topk(topk)
    topk_str_arr = []
    for k, obj_idx in enumerate(topk_idx):
        topk_str_arr.append('{} ({:.2f}%)'.format(
            weights[mdl].meta["categories"][obj_idx], 
            100 * topk_scores[k])
        )
    top_class = topk_str_arr.pop(0)
    topk_str = ', '.join(topk_str_arr)

    pred_list = [
        top_class,
        topk_str
        # f"{weights['ResNet50'].meta["categories"][class_id]}: {100 * pred[class_id].item():.2f}%"
    ]
    return pred_list
    # predict_df.loc[len(predict_df)] = pred_list
    # return predict_df