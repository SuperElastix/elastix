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
#include "MyStandardResampler/elxMyStandardResampler.h"
#ifdef ELASTIX_USE_OPENCL
#  include "OpenCLResampler/elxOpenCLResampler.h"
#endif

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
using elx::GTestUtilities::CreateDefaultElastixObject;
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
  template <template <typename> class TResamplerTemplate>
  struct WithResampler
  {
    using ResamplerType = TResamplerTemplate<ElastixType<NDimension>>;

    static void
    Test_CreateTransformParametersMap_for_default_resampler(const ParameterMapType & expectedDerivedParameterMap)
    {
      SCOPED_TRACE(std::string("Function = ")
                     .append(__func__)
                     .append("\n  ResamplerType = ")
                     .append(typeid(ResamplerType).name()));

      const auto                                    newResampler = CheckNew<ResamplerType>();
      elx::ResamplerBase<ElastixType<NDimension>> & resampler = *newResampler;

      const auto elastixObject = CreateDefaultElastixObject<ElastixType<NDimension>>();

      // Note: SetElastix does not take or share the ownership of its argument!
      resampler.SetElastix(elastixObject);

      ParameterMapType actualParameterMap;
      resampler.CreateTransformParametersMap(actualParameterMap);

      const ParameterMapType expectedBaseParameterMap = { { "Resampler", { resampler.elxGetClassName() } },
                                                          { "DefaultPixelValue", { "0" } },
                                                          { "ResultImageFormat", { "mhd" } },
                                                          { "ResultImagePixelType", { "short" } },
                                                          { "CompressResultImage", { "false" } } };

      ExpectAllKeysUnique(expectedDerivedParameterMap, expectedBaseParameterMap);
      EXPECT_EQ(actualParameterMap, MakeMergedMap(expectedDerivedParameterMap, expectedBaseParameterMap));
    }
  };


  static void
  Test_CreateTransformParametersMap_for_default_resampler()
  {
    WithResampler<elx::MyStandardResampler>::Test_CreateTransformParametersMap_for_default_resampler({});
#ifdef ELASTIX_USE_OPENCL
    WithResampler<elx::OpenCLResampler>::Test_CreateTransformParametersMap_for_default_resampler(
      { { "OpenCLResamplerUseOpenCL", { "true" } } });
#endif
  }
};

} // namespace


GTEST_TEST(Resampler, CreateTransformParametersMapForDefaultResampler)
{
  WithDimension<2>::Test_CreateTransformParametersMap_for_default_resampler();
  WithDimension<3>::Test_CreateTransformParametersMap_for_default_resampler();
}
