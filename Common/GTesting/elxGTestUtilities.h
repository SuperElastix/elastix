/*=========================================================================
 *
 *  Copyright UMC Utrecht and contributors
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
#ifndef elxGTestUtilities_h
#define elxGTestUtilities_h

#include <itkOptimizerParameters.h>
#include <itkPoint.h>
#include <itkSize.h>
#include <itkSmartPointer.h>
#include <itkVector.h>

// GoogleTest header file:
#include <gtest/gtest.h>

#include <algorithm> // For generate_n.
#include <cassert>
#include <cfloat> // For DBL_MAX.
#include <random>

namespace elastix
{
namespace GTestUtilities
{

/// Expect that all keys of both specified maps are unique.
template <typename TMap>
void
ExpectAllKeysUnique(const TMap & map1, const TMap & map2)
{
  const auto endOfMap2 = map2.end();

  for (const auto & keyValuePair : map1)
  {
    EXPECT_EQ(map2.find(keyValuePair.first), endOfMap2);
  }
}


/// Makes a map by merging its two arguments together.
template <typename TMap>
TMap
MakeMergedMap(TMap map1, const TMap & map2)
{
  // Note: This for-loop should be equivalent to C++17 `map1.merge(TMap{map2});`
  for (const auto & keyValuePair : map2)
  {
    map1.insert(keyValuePair);
  }
  return map1;
}


/// Creates a default `ElastixTemplate<FixedImageType, MovingImageType>` object.
/// for unit testing purposes.
template <typename TElastix>
itk::SmartPointer<TElastix>
CreateDefaultElastixObject()
{
  using FixedImageType = typename TElastix::FixedImageType;
  using MovingImageType = typename TElastix::MovingImageType;

  const auto elastixObject = TElastix::New();

  elastixObject->SetConfiguration(elx::Configuration::New());

  const auto fixedImageContainer = elx::ElastixBase::DataObjectContainerType::New();
  fixedImageContainer->push_back(FixedImageType::New());
  elastixObject->SetFixedImageContainer(fixedImageContainer);

  const auto movingImageContainer = elx::ElastixBase::DataObjectContainerType::New();
  movingImageContainer->push_back(MovingImageType::New());
  elastixObject->SetMovingImageContainer(movingImageContainer);

  return elastixObject;
}


/// Returns an `OptimizerParameters` object, filled with pseudo random floating point numbers between the specified
/// minimum and maximum value.
inline itk::OptimizerParameters<double>
GeneratePseudoRandomParameters(const unsigned numberOfParameters, const double minValue, const double maxValue = 1.0)
{
  assert(minValue < maxValue);
  assert((maxValue - minValue) <= DBL_MAX);

  itk::OptimizerParameters<double> parameters(numberOfParameters);

  std::mt19937 randomNumberEngine;

  std::generate_n(parameters.begin(), numberOfParameters, [&randomNumberEngine, minValue, maxValue] {
    return std::uniform_real_distribution<>{ minValue, maxValue }(randomNumberEngine);
  });
  return parameters;
}

} // namespace GTestUtilities
} // namespace elastix


#endif
