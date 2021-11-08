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

// First include the header files to be tested:
#include "BSplineResampleInterpolator/elxBSplineResampleInterpolator.h"
#include "BSplineResampleInterpolatorFloat/elxBSplineResampleInterpolatorFloat.h"
#include "LinearResampleInterpolator/elxLinearResampleInterpolator.h"
#include "NearestNeighborResampleInterpolator/elxNearestNeighborResampleInterpolator.h"
#include "RayCastResampleInterpolator/elxRayCastResampleInterpolator.h"
#include "RDBSplineResampleInterpolator/elxRDBSplineResampleInterpolator.h" // For ReducedDimensionBSplineResampleInterpolator.

#include "elxElastixTemplate.h"
#include "elxGTestUtilities.h"
#include "../Core/Main/GTesting/elxCoreMainGTestUtilities.h"

// ITK header file:
#include <itkImage.h>

// GoogleTest header file:
#include <gtest/gtest.h>

// Standard C++ header file:
#include <typeinfo>


using ParameterValuesType = itk::ParameterFileParser::ParameterValuesType;
using ParameterMapType = itk::ParameterFileParser::ParameterMapType;


// Using-declarations:
using elx::CoreMainGTestUtilities::CheckNew;
using elx::GTestUtilities::ExpectAllKeysUnique;
using elx::GTestUtilities::MakeMergedMap;


namespace
{

template <unsigned NDimension>
using ElastixType = elx::ElastixTemplate<itk::Image<float, NDimension>, itk::Image<float, NDimension>>;


// All tests specific to a dimension.
template <unsigned NDimension>
struct WithDimension
{
  template <template <typename> class TInterpolatorTemplate>
  struct WithInterpolator
  {
    using InterpolatorType = TInterpolatorTemplate<ElastixType<NDimension>>;

    static void
    Test_CreateTransformParametersMap_for_default_interpolator(const ParameterMapType & expectedDerivedParameterMap)
    {
      SCOPED_TRACE(std::string("Function = ")
                     .append(__func__)
                     .append("\n  InterpolatorType = ")
                     .append(typeid(InterpolatorType).name()));

      const auto                                                     newInterpolator = CheckNew<InterpolatorType>();
      const elx::ResampleInterpolatorBase<ElastixType<NDimension>> & interpolator = *newInterpolator;

      ParameterMapType actualParameterMap;
      interpolator.CreateTransformParametersMap(actualParameterMap);

      const ParameterMapType expectedBaseParameterMap = { { "ResampleInterpolator",
                                                            { interpolator.elxGetClassName() } } };

      ExpectAllKeysUnique(expectedDerivedParameterMap, expectedBaseParameterMap);
      EXPECT_EQ(actualParameterMap, MakeMergedMap(expectedDerivedParameterMap, expectedBaseParameterMap));
    }
  };


  static void
  Test_CreateTransformParametersMap_for_default_interpolator()
  {
    using namespace elx;

    const std::string expectedFinalBSplineInterpolationOrderKey = "FinalBSplineInterpolationOrder";
    const std::string expectedZero = "0";

    WithInterpolator<BSplineResampleInterpolator>::Test_CreateTransformParametersMap_for_default_interpolator(
      { { expectedFinalBSplineInterpolationOrderKey, { "3" } } });
    WithInterpolator<BSplineResampleInterpolatorFloat>::Test_CreateTransformParametersMap_for_default_interpolator(
      { { expectedFinalBSplineInterpolationOrderKey, { "3" } } });
    WithInterpolator<LinearResampleInterpolator>::Test_CreateTransformParametersMap_for_default_interpolator({});
    WithInterpolator<NearestNeighborResampleInterpolator>::Test_CreateTransformParametersMap_for_default_interpolator(
      {});

    const auto skippedTest = [expectedZero] {
      // Note: The following crashes when trying to retrieve "PreParameters" by `m_PreTransform->GetParameters()`.
      WithInterpolator<RayCastResampleInterpolator>::Test_CreateTransformParametersMap_for_default_interpolator(
        { { "FocalPoint", ParameterValuesType(NDimension, expectedZero) },
          { "PreParameters", { expectedZero } },
          { "Threshold", { expectedZero } } });
    };
    (void)skippedTest;

    WithInterpolator<ReducedDimensionBSplineResampleInterpolator>::
      Test_CreateTransformParametersMap_for_default_interpolator(
        { { expectedFinalBSplineInterpolationOrderKey, { "1" } } });
  }
};

} // namespace


GTEST_TEST(ResampleInterpolator, CreateTransformParametersMapForDefaultInterpolator)
{
  WithDimension<2>::Test_CreateTransformParametersMap_for_default_interpolator();
  WithDimension<3>::Test_CreateTransformParametersMap_for_default_interpolator();
}
