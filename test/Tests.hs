module Main where

import Test.HUnit
import VigraRF
import Data.Word
import System.Random

main = runTestTT tests

tests = TestCase testMultiDimCheckBoard

testMultiDimCheckBoard :: Assertion
testMultiDimCheckBoard = do
  let (f,l) = multiDimCheckBoard (mkStdGen 0) 5 10000
  rf <- learnRandomForest defaultRandomForestOptions (take 9000 f) (take 9000 l)
  pred <- predictRandomForest rf (drop 9000 f)
  let result = zipWith (\a b -> if a !! (fromIntegral b) > 0.5 then 1 else 0) pred (drop 9000 l)
      numRight = sum result
  assertBool "Bad test score of random forest" (numRight > 9000)


multiDimCheckBoard :: RandomGen g => g -> Int -> Int -> (Features,Labels)
multiDimCheckBoard gen numSamples dims =
  let features = genFeatures gen
      labels   = map label features
  in (features, labels)
 where
  genInfFeatures gen = let (gen1,gen2) = split gen
                    in  (genFeature gen1):(genInfFeatures gen2)
  genFeatures = take numSamples . genInfFeatures

  genInfFeature gen = let (i,nextGen) = randomR (-10000.0 :: Double,10000.0 :: Double) gen 
                       in  (i / 1000.0):(genInfFeature nextGen)
  genFeature = take dims . genInfFeature
  label :: [Double] -> Word
  label = boolToUInt . xor . map (odd . floor)
  xor :: [Bool] -> Bool
  xor [] = False
  xor (a:as) = if a then not (xor as) else (xor as)
  boolToUInt a = if a then 1 else 0
