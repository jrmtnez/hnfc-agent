def get_numeric_rating(class_values, text_rating, binary_classifier=False):
    if binary_classifier:
        if class_values.index(text_rating) == len(class_values) - 1:
            return 1
        else:
            return 0
    else:
        return class_values.index(text_rating)


def get_textual_rating(class_values, numeric_rating):
    return class_values[numeric_rating]


def get_number_of_classes(binary_classifier, class_values):
    if binary_classifier:
        return 2
    return len(class_values)
