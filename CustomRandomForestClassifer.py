import random

"""
    This is a simple implementation of the Random Forest Classifier without using any libraries. In itself decision
    trees are not very accurate. But a collection of these decision trees forms the Random Forest classifier which
    is very powerful.
    
    Constructing a Random Forest Classifier,
    
        1) Build a function to generate decision tree given data.
        2) Partition data to required number of partitions based on number of trees in the forest.
        3) Call the build decision tree function with each of these partitions to generate the forest of decision trees.
    
    Classification,
        
        1) Build a classification function that classifies the test data using a single decision tree.
        2) Call this function with every tree in the forest and the test data.
        3) Return the most voted label as classification result.

"""

training_headers = None

def readInputFromCSV(absoluteFilePath):
    """
    Read the dataset from the the specified CSV file.
    """
    global training_headers

    dataset = []

    fileObject = open(absoluteFilePath, "r")

    line = fileObject.readline()
    training_headers = line[:-1].split(",")

    line = fileObject.readline()

    while line:
        dataset.append(line[:-1].split(","))
        line = fileObject.readline()

    return dataset

class DecisionTreeNode:
    """
    Class that is used to represent each node in a decision tree.
        If it is a decision node, decision_tuple, true_subtree and false_subtree would NOT be None and leaf_value would be None.
        If it is a leaf node, decision_tuple, true_subtree and false_subtree would be None and leaf_value would NOT be None.
    """

    def __init__(self, decision_tuple,  true_subtree, false_subtree, leaf_value = None):
        """
        Constructor for the decision tree node.
        """

        # The decision_tuple is a list of the type [feature, value], indicating that decision must be based on the feature and value specified.
        self.decision_tuple = decision_tuple

        # This is class variable used to represent the subset of the data that satisfy the decision_tuple.
        self.true_subtree = true_subtree

        # This is class variable used to represent the subset of the data that do NOT satisfy the decision_tuple.
        self.false_subtree = false_subtree

        # This is class variable that indicates leaf node and is used to represent result of the classification.
        self.leaf_value = leaf_value



def setup_class_labels(dataset_rows):
    """
    This function takes in a subset of the data and returns a dictionary with all the labels as keys and number of data rows
    in each of those labels as their corresponding values.
    """
    classlabel_dictionary = {}

    for data_row in dataset_rows:
        current_class_label = data_row[-1]
        if not(current_class_label in classlabel_dictionary):
            classlabel_dictionary[current_class_label] = 0
        classlabel_dictionary[current_class_label] += 1

    return classlabel_dictionary

def compute_gini_impurity(dataset_rows):
    """
    This function is used to compute the Gini value of the subset of data as mentioned before and returns this value.
    """
    class_labels = setup_class_labels(dataset_rows)
    number_of_data_rows = len(dataset_rows)
    gini_impurity = 0

    for label in class_labels:
        probability_of_label = float(class_labels[label]) / number_of_data_rows
        gini_impurity += (probability_of_label - (probability_of_label**2))

    return gini_impurity

def check_feature_value_with_decision_tuple(dataset_row, decision_tuple):
    """
    This function compares the feature value of the input data row and compares it with the decision tuple value in order
    tp check whether the row satisfies the decision or not.
    """
    feature_index = decision_tuple[0]
    unique_feature_value = decision_tuple[1]

    dataset_feature_value = dataset_row[feature_index]

    try :
        dataset_feature_value = float(dataset_feature_value)
        unique_feature_value = float(unique_feature_value)

        return (dataset_feature_value >= unique_feature_value)

    except :
        return (dataset_feature_value == unique_feature_value)

def separate_data_using_decisiontuple(dataset_rows, decision_tuple):
    """
    This function is used to split the input into true and false data based on the decison tuple. If a row satisfies the
    decision, it is appended to true data, else it is appended to the false data.
    """
    true_dataset_rows = []
    false_dataset_rows = []
    for dataset_row in dataset_rows:
        if check_feature_value_with_decision_tuple(dataset_row, decision_tuple) == True:
            true_dataset_rows.append(dataset_row)
        else:
            false_dataset_rows.append(dataset_row)

    return [true_dataset_rows, false_dataset_rows]

def compute_information_gain(true_dataset_rows, false_dataset_rows, current_impurity):
    """
    Given the true data, false data and current impurity, this function computes the information gain as mentioned.
    """
    probability_of_true_row = float(len(true_dataset_rows)) / (len(true_dataset_rows) + len(false_dataset_rows))
    probability_of_false_row = 1 - probability_of_true_row

    information_gain = current_impurity - (probability_of_true_row * compute_gini_impurity(true_dataset_rows)) - (probability_of_false_row * compute_gini_impurity(false_dataset_rows))

    return information_gain


def get_best_split(dataset_rows):
    """
    This function is used to compute the best split according to the data subset passed.
    To do this,
        1) Compute the Gini Impurity of the current data.
        2) Compute the Information gain of each unique value of every feature to determine the best splitting decision.
    """

    best_information_gain = 0

    best_decision_tuple = None

    current_impurity = compute_gini_impurity(dataset_rows)

    number_of_data_rows = len(dataset_rows)

    number_of_features = len(dataset_rows[0]) - 1

    for feature_index in range(0, number_of_features):
        feature_values = []

        for dataset_row_counter in range(0, number_of_data_rows):
            feature_values.append(dataset_rows[dataset_row_counter][feature_index])
        feature_values = set(feature_values)

        for unique_feature_value in feature_values:

            current_decision_tuple = (feature_index, unique_feature_value)

            result_of_separation = separate_data_using_decisiontuple(dataset_rows, current_decision_tuple)
            true_dataset_rows, false_dataset_rows = result_of_separation[0], result_of_separation[1]

            current_information_gain = compute_information_gain(true_dataset_rows, false_dataset_rows, current_impurity)

            if current_information_gain >= best_information_gain:
                best_information_gain = current_information_gain
                best_decision_tuple = current_decision_tuple

    return [best_information_gain, best_decision_tuple]

def print_tree(node, spacing=""):
    """
    A function used to visualize the decision tree recursively
    """

    if (node.leaf_value != None):
        print (spacing + "Class : ", str(node.leaf_value))
        return

    print (spacing + "Is " + training_headers[node.decision_tuple[0]] + " : " + str(node.decision_tuple[1]))

    print (spacing + 'True:')
    print_tree(node.true_subtree, spacing + "\t")

    print (spacing + 'False:')
    print_tree(node.false_subtree, spacing + "\t")


def construct_decision_tree(dataset_rows):
    """
    This function is used to actually construct the decision tree in a recursive manner,
        1) Find Best Split.
        2) If info gain = 0, no more splits to make and hence is a leaf node and we return the leaf prediction.
        3) Else, split the data according to the computed best split and then recursively call the method with the true
        and false subtrees.
    """

    split_result = get_best_split(dataset_rows)

    information_gain = split_result[0]
    decision_tuple = split_result[1]

    if information_gain == 0:
        classification_result = ""
        for key in setup_class_labels(dataset_rows):
            classification_result += key + " or "

        classification_result = " ".join((classification_result.split(" "))[0:-2])
        return DecisionTreeNode(None, None, None, classification_result)

    separation_result = separate_data_using_decisiontuple(dataset_rows, decision_tuple)
    true_dataset_rows, false_dataset_rows = separation_result[0], separation_result[1]

    true_subtree = construct_decision_tree(true_dataset_rows)

    false_subtree = construct_decision_tree(false_dataset_rows)

    return DecisionTreeNode(decision_tuple, true_subtree, false_subtree)


def classify_data_row(classification_row, decision_tree_node):
    """
    This function is used to recursively classify data into a label given a node in the decision tree
    """
    if decision_tree_node.decision_tuple == None:
        return decision_tree_node.leaf_value

    if check_feature_value_with_decision_tuple(classification_row, decision_tree_node.decision_tuple) == True:
        return classify_data_row(classification_row, decision_tree_node.true_subtree)
    else:
        return classify_data_row(classification_row, decision_tree_node.false_subtree)


def partition_dataset(number_of_partitions, dataset):
    """
    This function is used to partition the dataset randomly into specified number of partitions.
    """
    partioned_dataset = []
    size_of_dataset = len(dataset)

    if number_of_partitions > size_of_dataset:
        print("size_of_partition > size_of_dataset; Cannot be partitioned !")
        return

    for i in range(0, number_of_partitions):
        partioned_dataset.append([])

    random.shuffle(dataset)

    for i in range(0, size_of_dataset):
        partition_number = i % number_of_partitions
        partioned_dataset[partition_number].append(dataset[i])

    return partioned_dataset


def build_random_forest(number_of_trees, dataset):
    """
    This function is used to build the random forest with the specified number of trees by individually building each
    decision tree with a data partition.
    """
    forest = []

    size_of_dataset = len(dataset)

    if size_of_dataset < number_of_trees:
        print("size_of_dataset < number_of_trees; Cannot generate forest!")
        return

    dataset_partitions = partition_dataset(number_of_trees, dataset)

    for partition in dataset_partitions:
        forest.append(construct_decision_tree(partition))

    return forest

def random_forest_classification(classification_row, forest):
    """
    This function calls the classification method on each decision tree in the forest, computes the maximum voted
    label and returns this label.
    """
    classification_result_dictionary = {}

    for tree in forest:
        result = classify_data_row(classification_row, tree)

        if not(result in classification_result_dictionary):
            classification_result_dictionary[result] = 0

        classification_result_dictionary[result] += 1

    max, max_key = 0, None
    for key in classification_result_dictionary.keys():
        print("Keys : ", key, "  |  Value : ", classification_result_dictionary[key])
        if classification_result_dictionary[key] > max:
            max = classification_result_dictionary[key]
            max_key = key

    return max_key


if __name__ == '__main__':
    data = readInputFromCSV("sample.csv")

    forest = build_random_forest(2, data)

    test_data = []
	
    classification_label = random_forest_classification(testData)

    print("Result of Classification : , class_label)
