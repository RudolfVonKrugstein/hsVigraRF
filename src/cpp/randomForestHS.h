/**
 * C++/C part for interfacing the random forest from vigra with haskell.
 */

#include <vigra/random_forest.hxx>

// We don't allow any label or feature type. We are strict about it!
typedef vigra::RandomForest<unsigned int>  RandomForest;
typedef vigra::MultiArray<2u,unsigned int> LabelArray;
typedef vigra::MultiArray<2u,double>       FeatureArray;
typedef vigra::MultiArray<2u,double>       PredictionArray;

extern "C" {

// Functions to create label and feature arrays
LabelArray* createLabelArray(unsigned int size);
void deleteLabelArray(LabelArray* array);
FeatureArray* createFeatureArray(unsigned int num_data, unsigned int num_features);
void deleteFeatureArray(FeatureArray* array);
void setLabelArrayValue(LabelArray* array, unsigned int dataid, unsigned int val);
void setFeatureArrayValue(FeatureArray* array, unsigned int dataid, unsigned int featureId, double val);
double getPredictionArrayValue(PredictionArray* array, unsigned int dataid, unsigned int labelId);
void deletePredictionArray(PredictionArray* array);
RandomForest* createRandomForest(int treeCount, int mtry, int min_split_node_size, int training_set_size, float training_set_proportions, int sample_with_replacement, int sample_classes_individually);
void deleteRandomForest(RandomForest* rf);
double learnRandomForest(RandomForest* rf
                        ,FeatureArray *trainData
                        ,LabelArray *trainLabels
                        ,int printProgress);
PredictionArray* predictRandomForest(RandomForest* rf
                        ,FeatureArray* testData);
unsigned int getNumClasses(RandomForest* rf);
}
