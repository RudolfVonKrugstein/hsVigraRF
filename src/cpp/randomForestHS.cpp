/**
 * C++/C part for interfacing the random forest from vigra with haskell.
 */

#include <vigra/random_forest.hxx>
#include "randomForestHS.h"
#include <stdio.h>

// Functions to create label and feature arrays
LabelArray* createLabelArray(unsigned int size) {
  printf("Creating label array\n");
  return new LabelArray(vigra::MultiArrayShape<2>::type(size,1));
}

void deleteLabelArray(LabelArray* array) {
  delete array;
}

FeatureArray* createFeatureArray(unsigned int num_data, unsigned int num_features) {
  printf("Creating feature array\n");
  return new FeatureArray(vigra::MultiArrayShape<2>::type(num_data, num_features));
}

void deleteFeatureArray(FeatureArray* array) {
  delete array;
}

void deletePredictionArray(PredictionArray* array) {
  delete array;
}

void setLabelArrayValue(LabelArray* array, unsigned int dataid, unsigned int val) {
  (*array)[vigra::MultiArrayShape<2>::type(dataid,0)] = val;
}

void setFeatureArrayValue(FeatureArray* array, unsigned int dataid, unsigned int featureId, double val) {
  (*array)[vigra::MultiArrayShape<2>::type(dataid,featureId)] = val;
}

double getPredictionArrayValue(PredictionArray* array, unsigned int dataid, unsigned int labelId) {
  return (*(PredictionArray*)array)[vigra::MultiArrayShape<2>::type(dataid,labelId)];
}

// sample_with_replacement and sample_classes_individually should be of type bool
// But there is no CBool in haskell so we use int (CInt)
RandomForest* createRandomForest(int treeCount, int mtry, int min_split_node_size, int training_set_size, float training_set_proportions, int sample_with_replacement, int sample_classes_individually) {
  printf("Creating random forest\n");
  vigra::RandomForestOptions options;
  options.sample_with_replacement(sample_with_replacement)
         .tree_count(treeCount)
         .min_split_node_size(min_split_node_size);
  if (mtry > 0)
    options.features_per_node(mtry);
  if (training_set_size != 0)
    options.samples_per_tree(training_set_size);
  else
    options.samples_per_tree(training_set_proportions);
  if (sample_classes_individually)
    options.use_stratification(vigra::RF_EQUAL);

  printf("Options done, calling new\n");
  RandomForest *rf = new RandomForest(options);
  printf("Returning rf");
  return rf;
}

void deleteRandomForest(RandomForest* rf) {
  delete rf;
}

double learnRandomForest(RandomForest* rf
                        ,FeatureArray *trainData
                        ,LabelArray *trainLabels) {
  using namespace vigra::rf;
  printf("Learning\n");
  visitors::OOB_Error oob_v;
  rf->learn(*trainData, *trainLabels, visitors::create_visitor(oob_v));
  return oob_v.oob_breiman;
}

PredictionArray* predictRandomForest(RandomForest* rf
                         ,FeatureArray* testData) {
  PredictionArray* res = new PredictionArray(vigra::MultiArrayShape<2>::type(testData->shape(0), rf->ext_param_.class_count_));
  printf("Predicting\n");
  rf->predictProbabilities(*testData, *res);
  return res;
}

unsigned int getNumClasses(RandomForest* rf) {
  return rf->ext_param_.class_count_;
}
