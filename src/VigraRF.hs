{-# LANGUAGE ForeignFunctionInterface #-}

-- |
-- Module      : VigraRF
-- Copyright   : (c) Nathan Huesken 2013
-- License     : MIT

module VigraRF
(
  RandomForest
, RandomForestOptions(..)
, defaultRandomForestOptions
, Label
, Labels
, Feature
, Features
, Predictions
, learnRandomForest
, predictRandomForest  
) where

import Data.Word
import Data.Functor
import Foreign
import Foreign.C.Types


-- | Options to pass to the random forest
data RandomForestOptions = RandomForestOptions {
  treeCount                   :: Word -- ^ The number of trees in the random forest
, features_per_node           :: Int  -- ^ The number of features that should be considered for splitting at each node. (-1) for SQRT of total number of features.
, min_split_node_size         :: Word -- ^ The minimum number of samples in a node causing it to be split
, training_set_size           :: Word -- ^ 
, training_set_proportions    :: Double
, sample_with_replacement     :: Bool -- ^ Sample from training population with or without replacement?
, sample_classes_individually :: Bool -- ^ Take sampels for each class induvidualy?
, print_progress              :: Bool -- ^ Should the progress be printed ot stdout
}
-- | Default options for reandom forest
defaultRandomForestOptions :: RandomForestOptions
defaultRandomForestOptions = RandomForestOptions {
  treeCount                   = 255
, features_per_node           = -1
, min_split_node_size         = 1
, training_set_size           = 0
, training_set_proportions    = 1.0
, sample_with_replacement     = True
, sample_classes_individually = False
, print_progress              = True}

-- | Label type
type Label    = Word
type Labels   = [Label]
-- | Feature type
type Feature  = [Double]
type Features = [Feature]
-- | Prediction type (output of the random forest)
--   Its a list of lists of probabilities.
--   The probability for the n-th class in the m-th sample is 'pred !! m !! n'
type Predictions = [[Double]]

-- High level functions
-- | Create and train a random forest
learnRandomForest :: RandomForestOptions -> Features -> Labels -> IO RandomForest
learnRandomForest os fs ls = do
  featureArray <- createFeatureArray fs
  labelArray   <- createLabelArray ls
  randomForest <- createRandomForest os
  withForeignPtr featureArray $
    \fa -> withForeignPtr labelArray $
      \la -> withForeignPtr randomForest $
        \rf -> c_learnRandomForest rf fa la (if print_progress os then 1 else 0)
  return randomForest

-- | Predict labels using a random forest with the given features
predictRandomForest :: RandomForest -> Features -> IO Predictions
predictRandomForest randomForest fs = do
  featureArray  <- createFeatureArray fs
  unsafeArray   <- withForeignPtr featureArray $
                   \fa -> withForeignPtr randomForest $
                     \rf -> c_predictRandomForest rf fa
  predArray   <- newForeignPtr c_deletePredictionArray unsafeArray
  classes     <- getNumClasses randomForest
  let classIds = [0..(classes-1)] :: [Word]
      dataIds  = [0..(fromIntegral $ length fs - 1)] :: [Word]
  mapM (\i -> mapM (\j -> realToFrac <$> getPredictionArrayValue predArray i j) classIds) dataIds

-- From here are the ffi functions
-- | Foreign label array type
data CLabelArray      = CLabelArray
type LabelArray       = ForeignPtr CLabelArray
-- | Foreign feature array type
data CFeatureArray    = CFeatureArray
type FeatureArray     = ForeignPtr CFeatureArray
-- | Foreign prediction array type
data CPredictionArray = CPredictionArray
type PredictionArray  = ForeignPtr CPredictionArray
-- | Foreign type for random forest
data CRandomForest    = CRandomForest
type RandomForest     = ForeignPtr CRandomForest


-- All foreign calls with "non" foreign calls that take care of memory menangment
foreign import ccall "randomForestHS.h createLabelArray" c_createLabelArray :: CUInt -> IO (Ptr CLabelArray)
foreign import ccall "randomForestHS.h &deleteLabelArray" c_deleteLabelArray :: FunPtr (Ptr CLabelArray -> IO ())
foreign import ccall "randomForestHS.h createFeatureArray" c_createFeatureArray :: CUInt -> CUInt -> IO (Ptr CFeatureArray)
foreign import ccall "randomForestHS.h &deleteFeatureArray" c_deleteFeatureArray :: FunPtr (Ptr CFeatureArray -> IO ())
foreign import ccall "randomForestHS.h &deletePredictionArray" c_deletePredictionArray :: FunPtr (Ptr CPredictionArray -> IO ())

-- foreign functions to get and set values
foreign import ccall "randomForest.h setLabelArrayValue" c_setLabelArrayValue :: Ptr CLabelArray -> CUInt -> CUInt -> IO ()
setLabelArrayValue :: LabelArray -> Word -> Word -> IO ()
setLabelArrayValue a x v = withForeignPtr a (\pa -> c_setLabelArrayValue pa (fromIntegral x) (fromIntegral v))

foreign import ccall "randomForest.h setFeatureArrayValue" c_setFeatureArrayValue :: Ptr CFeatureArray -> CUInt -> CUInt -> CDouble -> IO ()
setFeatureArrayValue :: FeatureArray -> Word -> Word -> Double -> IO ()
setFeatureArrayValue a x y v = withForeignPtr a (\pa -> c_setFeatureArrayValue pa (fromIntegral x) (fromIntegral y) (realToFrac v))

foreign import ccall "randomForest.h getPredictionArrayValue" c_getPredictionArrayValue :: Ptr CPredictionArray -> CUInt -> CUInt -> IO CDouble
getPredictionArrayValue :: PredictionArray -> Word -> Word -> IO Float
getPredictionArrayValue a x y = realToFrac <$> withForeignPtr a (\pa -> c_getPredictionArrayValue pa (fromIntegral x) (fromIntegral y))


-- Memory managed versions, also filling values
createLabelArray :: Labels -> IO LabelArray
createLabelArray ls = do
  array <- c_createLabelArray (fromIntegral . length $ ls) >>= newForeignPtr c_deleteLabelArray
  mapM_ (\(l,i) -> setLabelArrayValue array i l) $ zip ls [0 :: Word ..]
  return array

createFeatureArray :: Features -> IO FeatureArray
createFeatureArray ls = do
  array <- c_createFeatureArray (fromIntegral . length $ ls)  (fromIntegral . length . head $ ls) >>= newForeignPtr c_deleteFeatureArray
  mapM_ (\(ls2,i) -> mapM_ (\(l,j) -> setFeatureArrayValue array i j l) $ zip ls2 [0 :: Word ..]) $ zip ls [0 :: Word ..]
  return array

-- foreign import to create random forest
foreign import ccall "randomForestHS.h createRandomForest" c_createRandomForest :: CInt -> CInt -> CInt -> CInt -> CFloat -> CInt -> CInt -> IO (Ptr CRandomForest)
foreign import ccall "randomForestHS.h &deleteRandomForest" c_deleteRandomForest :: FunPtr (Ptr CRandomForest -> IO ())

createRandomForest :: RandomForestOptions -> IO RandomForest
createRandomForest o = do
  unsafePtr <- c_createRandomForest (fromIntegral . treeCount                   $ o)
                                    (fromIntegral . features_per_node           $ o)
                                    (fromIntegral . min_split_node_size         $ o)
                                    (fromIntegral . training_set_size           $ o)
                                    (realToFrac   . training_set_proportions    $ o)
                                    (if sample_with_replacement o then 1 else 0)
                                    (if sample_classes_individually o then 1 else 0)
  newForeignPtr c_deleteRandomForest unsafePtr

foreign import ccall "randomForestHS.h getNumClasses" c_getNumClasses :: Ptr CRandomForest -> IO CUInt
getNumClasses :: RandomForest -> IO Word
getNumClasses rf = fromIntegral <$> withForeignPtr rf c_getNumClasses

-- foreign import to learn and predict random forest
foreign import ccall "randomForestHS.h learnRandomForest" c_learnRandomForest :: Ptr CRandomForest -> Ptr CFeatureArray -> Ptr CLabelArray -> CInt -> IO CFloat
foreign import ccall "randomForestHS.h predictRandomForest" c_predictRandomForest :: Ptr CRandomForest -> Ptr CFeatureArray -> IO (Ptr CPredictionArray)
