from cross_arch_evaluator import CrossArchEvaluator




def main():
    # load dataset
    images_train = None
    labels_train = None
    dst_train = TensorDataset(images_train, labels_train)

    cross_arch_eval = CrossArchEvaluator()

    result = cross_arch_eval.evaluat()
    print("final result is", result)





if __name__ == '__main__':
    main()