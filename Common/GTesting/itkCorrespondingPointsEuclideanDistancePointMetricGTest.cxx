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

// First include the header file to be tested:
#include "CorrespondingPointsEuclideanDistanceMetric/itkCorrespondingPointsEuclideanDistancePointMetric.h"
#include "itkAdvancedTranslationTransform.h"
#include "elxDefaultConstruct.h"
#include <itkPointSet.h>
#include <gtest/gtest.h>

// The template to be tested.
using itk::CorrespondingPointsEuclideanDistancePointMetric;

// Checks if a default-constructed CorrespondingPointsEuclideanDistancePointMetric has the expected properties.
GTEST_TEST(CorrespondingPointsEuclideanDistancePointMetric, DefaultConstruct)
{
  using PointSetType = itk::PointSet<double>;
  using MetricType = CorrespondingPointsEuclideanDistancePointMetric<PointSetType, PointSetType>;
  const elx::DefaultConstruct<MetricType>                                       defaultConstructedMetric{};
  const itk::SingleValuedPointSetToPointSetMetric<PointSetType, PointSetType> & pointSetToPointSetMetric =
    defaultConstructedMetric;

  EXPECT_EQ(pointSetToPointSetMetric.GetFixedPointSet(), nullptr);
  EXPECT_EQ(pointSetToPointSetMetric.GetMovingPointSet(), nullptr);
  EXPECT_EQ(pointSetToPointSetMetric.GetFixedImageMask(), nullptr);
  EXPECT_EQ(pointSetToPointSetMetric.GetMovingImageMask(), nullptr);
  EXPECT_EQ(pointSetToPointSetMetric.GetTransform(), nullptr);
  EXPECT_TRUE(pointSetToPointSetMetric.GetUseMetricSingleThreaded());

  // Note: `pointSetToPointSetMetric.m_NumberOfPointsCounted` is not public, and does not have a Get member function to
  // test its value.
}


// Checks if CorrespondingPointsEuclideanDistancePointMetric has the expected result on a minimal point set pair.
GTEST_TEST(CorrespondingPointsEuclideanDistancePointMetric, MinimalPointSets)
{
  using PointSetType = itk::PointSet<double>;
  using PointsVectorContainerType = PointSetType::PointsVectorContainer;
  using MetricType = CorrespondingPointsEuclideanDistancePointMetric<PointSetType, PointSetType>;

  elx::DefaultConstruct<itk::AdvancedTranslationTransform<double, PointSetType::PointDimension>> transform{};
  elx::DefaultConstruct<PointsVectorContainerType> fixedPointsVectorContainer{};

  fixedPointsVectorContainer.resize(PointSetType::PointDimension);
  elx::DefaultConstruct<PointsVectorContainerType> movingPointsVectorContainer{};
  movingPointsVectorContainer.CastToSTLContainer() = fixedPointsVectorContainer.CastToSTLConstContainer();
  elx::DefaultConstruct<PointSetType> fixedPointSet{};
  fixedPointSet.SetPoints(&fixedPointsVectorContainer);
  elx::DefaultConstruct<PointSetType> movingPointSet{};
  movingPointSet.SetPoints(&movingPointsVectorContainer);
  elx::DefaultConstruct<MetricType> metric{};
  metric.SetFixedPointSet(&fixedPointSet);
  metric.SetMovingPointSet(&movingPointSet);
  metric.SetTransform(&transform);
  metric.Initialize();
  EXPECT_EQ(metric.GetValue(transform.GetParameters()), 0);
}
