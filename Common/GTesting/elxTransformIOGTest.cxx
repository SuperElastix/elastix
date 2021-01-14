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
#include "elxTransformIO.h"

#include "elxElastixMain.h" // For xoutManager.
#include "elxElastixTemplate.h"

#include "AdvancedAffineTransform/elxAdvancedAffineTransform.h"
#include "AdvancedBSplineTransform/elxAdvancedBSplineTransform.h"
#include "AffineDTITransform/elxAffineDTITransform.h"
#include "AffineLogStackTransform/elxAffineLogStackTransform.h"
#include "AffineLogTransform/elxAffineLogTransform.h"
#include "BSplineDeformableTransformWithDiffusion/elxBSplineTransformWithDiffusion.h"
#include "BSplineStackTransform/elxBSplineStackTransform.h"
#include "DeformationFieldTransform/elxDeformationFieldTransform.h"
#include "EulerStackTransform/elxEulerStackTransform.h"
#include "EulerTransform/elxEulerTransform.h"
#include "MultiBSplineTransformWithNormal/elxMultiBSplineTransformWithNormal.h"
#include "RecursiveBSplineTransform/elxRecursiveBSplineTransform.h"
#include "SimilarityTransform/elxSimilarityTransform.h"
#include "SplineKernelTransform/elxSplineKernelTransform.h"
#include "TranslationStackTransform/elxTranslationStackTransform.h"
#include "TranslationTransform/elxTranslationTransform.h"
#include "WeightedCombinationTransform/elxWeightedCombinationTransform.h"

#include <itkImage.h>
#include <itkAffineTransform.h>
#include <itkBSplineTransform.h>
#include <itkEuler2DTransform.h>
#include <itkEuler3DTransform.h>
#include <itkSimilarity2DTransform.h>
#include <itkSimilarity3DTransform.h>
#include <itkTranslationTransform.h>

#include <typeinfo>
#include <type_traits> // For is_same

#include <gtest/gtest.h>

using ParameterValuesType = itk::ParameterFileParser::ParameterValuesType;
using ParameterMapType = itk::ParameterFileParser::ParameterMapType;

namespace
{

template <unsigned NDimension>
using ElastixType = elx::ElastixTemplate<itk::Image<float, NDimension>, itk::Image<float, NDimension>>;

template <std::size_t N>
using ExpectedParameters = std::array<double, N>;


// All tests specific to a dimension.
template <unsigned NDimension>
struct WithDimension
{
  template <template <typename> typename TElastixTransform>
  struct WithElastixTransform
  {
    using ElastixTransformType = TElastixTransform<ElastixType<NDimension>>;

    template <typename TExpectedCorrespondingItkTransform>
    static void
    Expect_CorrespondingItkTransform()
    {
      const auto elxTransform = ElastixTransformType::New();
      const auto itkTransform = elx::TransformIO::CreateCorrespondingItkTransform(*elxTransform);
      ASSERT_NE(itkTransform, nullptr);

      const auto & actualItkTransformTypeId = typeid(*itkTransform);
      const auto & expectedItkTransformTypeId = typeid(TExpectedCorrespondingItkTransform);
      ASSERT_EQ(std::string(actualItkTransformTypeId.name()), std::string(expectedItkTransformTypeId.name()));
      EXPECT_EQ(actualItkTransformTypeId, expectedItkTransformTypeId);
    }


    static void
    Expect_default_elastix_FixedParameters_empty()
    {
      const auto fixedParameters = ElastixTransformType::New()->GetFixedParameters();
      ASSERT_EQ(fixedParameters, vnl_vector<double>());
    }


    static void
    Expect_default_elastix_FixedParameters_are_all_zero()
    {
      const auto fixedParameters = ElastixTransformType::New()->GetFixedParameters();
      ASSERT_EQ(fixedParameters.size(), NDimension);
      ASSERT_EQ(fixedParameters, vnl_vector<double>(NDimension, 0.0));
    }


    static void
    Expect_default_elastix_Parameters_remain_the_same_when_set(const bool fixed)
    {
      SCOPED_TRACE(std::string("Function = ")
                     .append(__func__)
                     .append("\n  ElastixTransformType = ")
                     .append(typeid(ElastixTransformType).name())
                     .append("\n  fixed = ")
                     .append(elx::BaseComponent::BoolToString(fixed)));

      const auto transform = ElastixTransformType::New();
      const auto parameters = elx::TransformIO::GetParameters(fixed, *transform);
      elx::TransformIO::SetParameters(fixed, *transform, parameters);
      ASSERT_EQ(elx::TransformIO::GetParameters(fixed, *transform), parameters);
    }

    template <typename TExpectedCorrespondingItkTransform>
    static void
    Test_copying_default_parameters(const bool fixed)
    {
      SCOPED_TRACE(std::string("Function = ")
                     .append(__func__)
                     .append("\n  ElastixTransformType = ")
                     .append(typeid(ElastixTransformType).name())
                     .append("\n  TExpectedCorrespondingItkTransform = ")
                     .append(typeid(TExpectedCorrespondingItkTransform).name())
                     .append("\n  fixed = ")
                     .append(elx::BaseComponent::BoolToString(fixed)));

      const auto elxTransform = ElastixTransformType::New();
      SCOPED_TRACE(fixed);

      const auto itkTransform = elx::TransformIO::CreateCorrespondingItkTransform(*elxTransform);
      ASSERT_NE(itkTransform, nullptr);

      const auto parameters = elx::TransformIO::GetParameters(fixed, *elxTransform);
      elx::TransformIO::SetParameters(fixed, *itkTransform, parameters);

      ASSERT_EQ(elx::TransformIO::GetParameters(fixed, *itkTransform), parameters);
    }

    template <typename TExpectedCorrespondingItkTransform>
    static void
    Test_copying_parameters()
    {
      const auto elxTransform = ElastixTransformType::New();

      SCOPED_TRACE(elxTransform->elxGetClassName());

      const auto itkTransform = elx::TransformIO::CreateCorrespondingItkTransform(*elxTransform);
      ASSERT_NE(itkTransform, nullptr);

      const auto & actualItkTransformTypeId = typeid(*itkTransform);
      const auto & expectedItkTransformTypeId = typeid(TExpectedCorrespondingItkTransform);
      ASSERT_EQ(std::string(actualItkTransformTypeId.name()), std::string(expectedItkTransformTypeId.name()));
      EXPECT_EQ(actualItkTransformTypeId, expectedItkTransformTypeId);

      auto parameters = elxTransform->GetParameters();
      std::iota(std::begin(parameters), std::end(parameters), 1.0);
      std::for_each(std::begin(parameters), std::end(parameters), [](double & x) { x /= 8; });
      elxTransform->SetParameters(parameters);
      ASSERT_EQ(elxTransform->GetParameters(), parameters);

      auto fixedParameters = elxTransform->GetFixedParameters();
      std::iota(std::begin(fixedParameters), std::end(fixedParameters), 1.0);
      elxTransform->SetFixedParameters(fixedParameters);
      ASSERT_EQ(elxTransform->GetFixedParameters(), fixedParameters);

      itkTransform->SetParameters(parameters);
      itkTransform->SetFixedParameters(fixedParameters);

      ASSERT_EQ(itkTransform->GetParameters(), parameters);
      ASSERT_EQ(itkTransform->GetFixedParameters(), fixedParameters);
    }


    static void
    Test_CreateTransformParametersMap_for_default_transform(const ParameterMapType & expectedParameterMap)
    {
      SCOPED_TRACE(std::string("Function = ")
                     .append(__func__)
                     .append("\n  ElastixTransformType = ")
                     .append(typeid(ElastixTransformType).name()));

      const elx::xoutManager manager("", false, false);

      const auto elastixObject = ElastixType<NDimension>::New();

      // Note: SetConfiguration does not share ownership!
      const auto configuration = elx::Configuration::New();
      elastixObject->SetConfiguration(configuration);

      const auto imageContainer = elx::ElastixBase::DataObjectContainerType::New();
      imageContainer->push_back(itk::Image<float, NDimension>::New());
      elastixObject->SetFixedImageContainer(imageContainer);
      elastixObject->SetMovingImageContainer(imageContainer);

      const auto elxTransform = ElastixTransformType::New();
      elxTransform->SetElastix(elastixObject);

      // BeforeAll() appears necessary to for MultiBSplineTransformWithNormal
      // and AdvancedBSplineTransform to initialize the internal ITK
      // transform of the elastix transform, by calling
      // InitializeBSplineTransform(void)
      elxTransform->BeforeAll();

      // Overrule the default-constructors of BSplineTransformWithDiffusion
      // and DeformationFieldTransform which do SetReadWriteTransformParameters(false)
      elxTransform->SetReadWriteTransformParameters(true);

      ParameterMapType parameterMap;
      elxTransform->CreateTransformParametersMap(itk::OptimizerParameters<double>{}, &parameterMap);

      for (const auto & expectedParameter : expectedParameterMap)
      {
        const auto found = parameterMap.find(expectedParameter.first);
        const bool isExpectedKeyFound = found != end(parameterMap);

        SCOPED_TRACE("Expected key = " + expectedParameter.first);
        EXPECT_TRUE(isExpectedKeyFound);

        if (isExpectedKeyFound)
        {
          EXPECT_EQ(expectedParameter.second, found->second);
        }
      }

      const std::string expectedImageDimension{ char{ '0' + NDimension } };
      const std::string expectedInternalImagePixelType = "float";
      const std::string expectedZero = "0";
      const std::string expectedOne = "1";

      const std::pair<const std::string, ParameterValuesType> expectedTransformBaseParameters[] = {
        { "Direction", ParameterValuesType(NDimension * NDimension, expectedZero) },
        { "FixedImageDimension", { expectedImageDimension } },
        { "FixedInternalImagePixelType", { expectedInternalImagePixelType } },
        { "HowToCombineTransforms", { "Compose" } },
        { "Index", { ParameterValuesType(NDimension, expectedZero) } },
        { "InitialTransformParametersFileName", { "NoInitialTransform" } },
        { "MovingImageDimension", { expectedImageDimension } },
        { "MovingInternalImagePixelType", { expectedInternalImagePixelType } },
        { "NumberOfParameters", { expectedZero } },
        { "Origin", { ParameterValuesType(NDimension, expectedZero) } },
        { "Size", { ParameterValuesType(NDimension, expectedZero) } },
        { "Spacing", { ParameterValuesType(NDimension, expectedOne) } },
        { "Transform", { elxTransform->elxGetClassName() } },
        { "TransformParameters", {} },
        { "UseDirectionCosines", { "true" } }
      };

      for (const auto & expectedTransformBaseParameter : expectedTransformBaseParameters)
      {
        const auto found = parameterMap.find(expectedTransformBaseParameter.first);
        SCOPED_TRACE("Expected key = " + expectedTransformBaseParameter.first);
        const bool isExpectedKeyFound = found != end(parameterMap);
        EXPECT_TRUE(isExpectedKeyFound);

        if (isExpectedKeyFound)
        {
          EXPECT_EQ(found->second, expectedTransformBaseParameter.second);
        }
      }

      const std::size_t numberOfExpectedTransformBaseParameters{ GTEST_ARRAY_SIZE_(expectedTransformBaseParameters) };

      EXPECT_GE(parameterMap.size(), numberOfExpectedTransformBaseParameters);
      EXPECT_EQ(parameterMap.size() - numberOfExpectedTransformBaseParameters, expectedParameterMap.size());

      ParameterMapType missingParameters;

      for (const auto & parameter : parameterMap)
      {
        if (std::find(begin(expectedTransformBaseParameters), end(expectedTransformBaseParameters), parameter) ==
              end(expectedTransformBaseParameters) &&
            (expectedParameterMap.count(parameter.first) == 0))
        {
          EXPECT_TRUE(missingParameters.insert(parameter).second);
        }
      }
      EXPECT_EQ(missingParameters, ParameterMapType{});
    }

    static void
    Test_CreateTransformParametersMap_double_precision()
    {
      const elx::xoutManager manager("", false, false);

      // Use double 0.1111111111111111 as test value.
      constexpr auto testValue = 1.0 / 9.0;
      constexpr auto expectedPrecision = 6;
      EXPECT_EQ(expectedPrecision, std::ostringstream{}.precision());
      const auto expectedString = "0." + std::string(expectedPrecision, '1');

      const auto elastixObject = ElastixType<NDimension>::New();

      // Note: SetConfiguration does not share ownership!
      const auto configuration = elx::Configuration::New();
      elastixObject->SetConfiguration(configuration);

      const auto imageContainer = elx::ElastixBase::DataObjectContainerType::New();
      const auto image = itk::Image<float, NDimension>::New();
      image->SetOrigin(testValue);
      image->SetSpacing(testValue);
      imageContainer->push_back(image);

      elastixObject->SetFixedImageContainer(imageContainer);
      elastixObject->SetMovingImageContainer(imageContainer);

      const auto elxTransform = ElastixTransformType::New();
      elxTransform->SetElastix(elastixObject);
      elxTransform->BeforeAll();

      ParameterMapType                       parameterMap;
      const itk::OptimizerParameters<double> optimizerParameters(itk::Array<double>(vnl_vector<double>(2U, testValue)));
      elxTransform->CreateTransformParametersMap(optimizerParameters, &parameterMap);

      for (const auto key : { "TransformParameters", "Origin", "Spacing" })
      {
        const auto found = parameterMap.find(key);
        ASSERT_NE(found, end(parameterMap));
        for (const auto & actualString : found->second)
        {
          EXPECT_EQ(actualString, expectedString);
        }
      }
    }

    static void
    Test_CreateTransformParametersMap_SetUseAddition()
    {
      const elx::xoutManager manager("", false, false);

      const auto elastixObject = ElastixType<NDimension>::New();

      // Note: SetConfiguration does not share ownership!
      const auto configuration = elx::Configuration::New();
      elastixObject->SetConfiguration(configuration);

      const auto imageContainer = elx::ElastixBase::DataObjectContainerType::New();
      const auto image = itk::Image<float, NDimension>::New();
      imageContainer->push_back(image);

      elastixObject->SetFixedImageContainer(imageContainer);
      elastixObject->SetMovingImageContainer(imageContainer);

      const auto elxTransform = ElastixTransformType::New();
      elxTransform->SetElastix(elastixObject);
      elxTransform->BeforeAll();

      const auto expectHowToCombineTransforms = [&elxTransform](const char * const expectedParameterValue) {
        ParameterMapType parameterMap;
        elxTransform->CreateTransformParametersMap({}, &parameterMap);

        const auto found = parameterMap.find("HowToCombineTransforms");
        ASSERT_NE(found, end(parameterMap));
        EXPECT_EQ(found->second, ParameterValuesType{ expectedParameterValue });
      };

      expectHowToCombineTransforms("Compose");
      elxTransform->SetUseAddition(true);
      expectHowToCombineTransforms("Add");
      elxTransform->SetUseAddition(false);
      expectHowToCombineTransforms("Compose");
    }
  };


  static void
  Expect_default_AdvancedBSplineTransform_GetParameters_throws_ExceptionObject(const bool fixed)
  {
    const auto transform = elx::AdvancedBSplineTransform<ElastixType<NDimension>>::New();
    EXPECT_THROW(elx::TransformIO::GetParameters(fixed, *transform), itk::ExceptionObject);
  }


  template <template <typename> typename TElastixTransform, std::size_t NExpectedParameters>
  static void
  Expect_default_elastix_Parameters_equal(const ExpectedParameters<NExpectedParameters> & expectedParameters)
  {
    using ElastixTransformType = TElastixTransform<ElastixType<NDimension>>;

    SCOPED_TRACE(std::string("Function = ")
                   .append(__func__)
                   .append("\n  ElastixTransformType = ")
                   .append(typeid(ElastixTransformType).name()));

    const auto parameters = ElastixTransformType::New()->GetParameters();
    ASSERT_EQ(parameters, vnl_vector<double>(expectedParameters.data(), NExpectedParameters));
  }

  static void
  Test_CreateTransformParametersMap_for_default_transform()
  {
    // Converts the specified string to an std::vector<std::string>. Each vector element contains a character of the
    // specified string.
    const auto toVectorOfStrings = [](const std::string & str) {
      std::vector<std::string> result;
      for (const char ch : str)
      {
        result.push_back({ ch });
      }
      return result;
    };

    // Concatenates n times the specified string.
    const auto times = [](const unsigned n, const std::string & str) {
      std::string result;

      for (auto i = n; i > 0; --i)
      {
        result += str;
      }
      return result;
    };

    const std::string         expectedFalse("false");
    const std::string         expectedZero("0");
    const std::string         expectedOne("1");
    const ParameterValuesType expectedZeros(NDimension, "0");
    const ParameterValuesType expectedOnes(NDimension, "1");
    const auto expectedMatrixTranslation = toVectorOfStrings(times(NDimension, '1' + std::string(NDimension, '0')));
    const auto expectedGridDirection =
      toVectorOfStrings(times(NDimension - 1, '1' + std::string(NDimension, '0')) + '1');

    using namespace elx;

    WithElastixTransform<AdvancedAffineTransformElastix>::Test_CreateTransformParametersMap_for_default_transform(
      { { "CenterOfRotationPoint", expectedZeros } });
    WithElastixTransform<AdvancedBSplineTransform>::Test_CreateTransformParametersMap_for_default_transform(
      { { "BSplineTransformSplineOrder", { "3" } },
        { "GridDirection", expectedGridDirection },
        { "GridIndex", expectedZeros },
        { "GridOrigin", expectedZeros },
        { "GridSize", expectedZeros },
        { "GridSpacing", expectedOnes },
        { "UseCyclicTransform", { "false" } } });
    WithElastixTransform<AffineDTITransformElastix>::Test_CreateTransformParametersMap_for_default_transform(
      { { "CenterOfRotationPoint", expectedZeros }, { "MatrixTranslation", expectedMatrixTranslation } });
    WithElastixTransform<AffineLogStackTransform>::Test_CreateTransformParametersMap_for_default_transform(
      { { "CenterOfRotationPoint", ParameterValuesType(NDimension - 1, expectedZero) },
        { "NumberOfSubTransforms", { expectedZero } },
        { "StackOrigin", { expectedZero } },
        { "StackSpacing", { expectedOne } } });
    WithElastixTransform<AffineLogTransformElastix>::Test_CreateTransformParametersMap_for_default_transform(
      { { "CenterOfRotationPoint", expectedZeros }, { "MatrixTranslation", expectedMatrixTranslation } });

    const auto skippedTest = [] {
      // Appears to crash when internally calling GetSubTransform(0) while m_SubTransformContainer is still empty.
      WithElastixTransform<BSplineStackTransform>::Test_CreateTransformParametersMap_for_default_transform({});
    };
    (void)skippedTest;

    WithElastixTransform<BSplineTransformWithDiffusion>::Test_CreateTransformParametersMap_for_default_transform(
      { { "GridIndex", expectedZeros },
        { "GridOrigin", expectedZeros },
        { "GridSize", expectedZeros },
        { "GridSpacing", expectedOnes } });
    WithElastixTransform<DeformationFieldTransform>::Test_CreateTransformParametersMap_for_default_transform(
      { { "DeformationFieldInterpolationOrder", { expectedZero } } });
    WithElastixTransform<EulerStackTransform>::Test_CreateTransformParametersMap_for_default_transform(
      { { "CenterOfRotationPoint", ParameterValuesType(NDimension - 1, expectedZero) },
        { "NumberOfSubTransforms", { expectedZero } },
        { "StackOrigin", { expectedZero } },
        { "StackSpacing", { expectedOne } } });

    WithElastixTransform<EulerTransformElastix>::Test_CreateTransformParametersMap_for_default_transform(
      (NDimension == 3)
        ? ParameterMapType{ { "CenterOfRotationPoint", expectedZeros }, { "ComputeZYX", { expectedFalse } } }
        : ParameterMapType{ { "CenterOfRotationPoint", expectedZeros } });

    try
    {
      WithElastixTransform<MultiBSplineTransformWithNormal>::Test_CreateTransformParametersMap_for_default_transform(
        { { "GridIndex", expectedZeros },
          { "GridOrigin", expectedZeros },
          { "GridSize", expectedZeros },
          { "GridSpacing", expectedOnes },
          { "BSplineTransformSplineOrder", { "3" } },
          { "GridDirection", expectedGridDirection },
          { "MultiBSplineTransformWithNormalLabels", { itksys::SystemTools::GetCurrentWorkingDirectory() } } });
      EXPECT_FALSE("MultiBSplineTransformWithNormal::CreateTransformParametersMap is expected to throw an exception!");
    }
    catch (const itk::ExceptionObject & exceptionObject)
    {
      // TODO Avoid this exception!
      EXPECT_NE(std::strstr(exceptionObject.GetDescription(), "ERROR: Missing -labels argument!"), nullptr);
    }

    WithElastixTransform<RecursiveBSplineTransform>::Test_CreateTransformParametersMap_for_default_transform(
      { { "BSplineTransformSplineOrder", { "3" } },
        { "GridDirection", expectedGridDirection },
        { "GridIndex", expectedZeros },
        { "GridOrigin", expectedZeros },
        { "GridSize", expectedZeros },
        { "GridSpacing", expectedOnes },
        { "UseCyclicTransform", { expectedFalse } } });
    WithElastixTransform<SimilarityTransformElastix>::Test_CreateTransformParametersMap_for_default_transform(
      { { "CenterOfRotationPoint", expectedZeros } });
    WithElastixTransform<SplineKernelTransform>::Test_CreateTransformParametersMap_for_default_transform(
      { { "FixedImageLandmarks", {} },
        { "SplineKernelType", { "unknown" } },
        { "SplinePoissonRatio", { "0.3" } },
        { "SplineRelaxationFactor", { expectedZero } } });
    WithElastixTransform<TranslationStackTransform>::Test_CreateTransformParametersMap_for_default_transform(
      { { "NumberOfSubTransforms", { expectedZero } },
        { "StackOrigin", { expectedZero } },
        { "StackSpacing", { expectedOne } } });
    WithElastixTransform<TranslationTransformElastix>::Test_CreateTransformParametersMap_for_default_transform({});
    WithElastixTransform<WeightedCombinationTransformElastix>::Test_CreateTransformParametersMap_for_default_transform(
      { { "NormalizeCombinationWeights", { expectedFalse } }, { "SubTransforms", {} } });
  }
};


template <template <typename> typename TElastixTransform>
void
Expect_default_elastix_FixedParameters_are_all_zero()
{
  WithDimension<2>::WithElastixTransform<TElastixTransform>::Expect_default_elastix_FixedParameters_are_all_zero();
  WithDimension<3>::WithElastixTransform<TElastixTransform>::Expect_default_elastix_FixedParameters_are_all_zero();
  WithDimension<4>::WithElastixTransform<TElastixTransform>::Expect_default_elastix_FixedParameters_are_all_zero();
}


template <template <typename> typename TElastixTransform>
void
Expect_default_elastix_FixedParameters_empty()
{
  WithDimension<2>::WithElastixTransform<TElastixTransform>::Expect_default_elastix_FixedParameters_empty();
  WithDimension<3>::WithElastixTransform<TElastixTransform>::Expect_default_elastix_FixedParameters_empty();
  WithDimension<4>::WithElastixTransform<TElastixTransform>::Expect_default_elastix_FixedParameters_empty();
}


template <template <typename> typename TElastixTransform>
void
Expect_default_elastix_Parameters_remain_the_same_when_set(const bool fixed)
{
  WithDimension<2>::WithElastixTransform<TElastixTransform>::Expect_default_elastix_Parameters_remain_the_same_when_set(
    fixed);
  WithDimension<3>::WithElastixTransform<TElastixTransform>::Expect_default_elastix_Parameters_remain_the_same_when_set(
    fixed);
  WithDimension<4>::WithElastixTransform<TElastixTransform>::Expect_default_elastix_Parameters_remain_the_same_when_set(
    fixed);
}

} // namespace


GTEST_TEST(TransformIO, CorrespondingItkTransform)
{
  WithDimension<2>::WithElastixTransform<elx::AdvancedAffineTransformElastix>::Expect_CorrespondingItkTransform<
    itk::AffineTransform<double, 2>>();
  WithDimension<3>::WithElastixTransform<elx::AdvancedAffineTransformElastix>::Expect_CorrespondingItkTransform<
    itk::AffineTransform<double, 3>>();
  WithDimension<4>::WithElastixTransform<elx::AdvancedAffineTransformElastix>::Expect_CorrespondingItkTransform<
    itk::AffineTransform<double, 4>>();

  WithDimension<2>::WithElastixTransform<elx::AdvancedBSplineTransform>::Expect_CorrespondingItkTransform<
    itk::BSplineTransform<double, 2>>();
  WithDimension<3>::WithElastixTransform<elx::AdvancedBSplineTransform>::Expect_CorrespondingItkTransform<
    itk::BSplineTransform<double, 3>>();

  WithDimension<2>::WithElastixTransform<elx::TranslationTransformElastix>::Expect_CorrespondingItkTransform<
    itk::TranslationTransform<double, 2>>();
  WithDimension<3>::WithElastixTransform<elx::TranslationTransformElastix>::Expect_CorrespondingItkTransform<
    itk::TranslationTransform<double, 3>>();
  WithDimension<4>::WithElastixTransform<elx::TranslationTransformElastix>::Expect_CorrespondingItkTransform<
    itk::TranslationTransform<double, 4>>();

  WithDimension<2>::WithElastixTransform<elx::SimilarityTransformElastix>::Expect_CorrespondingItkTransform<
    itk::Similarity2DTransform<double>>();
  WithDimension<3>::WithElastixTransform<elx::SimilarityTransformElastix>::Expect_CorrespondingItkTransform<
    itk::Similarity3DTransform<double>>();

  WithDimension<2>::WithElastixTransform<elx::EulerTransformElastix>::Expect_CorrespondingItkTransform<
    itk::Euler2DTransform<double>>();
  WithDimension<3>::WithElastixTransform<elx::EulerTransformElastix>::Expect_CorrespondingItkTransform<
    itk::Euler3DTransform<double>>();
}


GTEST_TEST(TransformIO, DefaultAdvancedBSplineTransformGetParametersThrowsExceptionObject)
{
  for (const bool fixed : { false, true })
  {
    WithDimension<2>::Expect_default_AdvancedBSplineTransform_GetParameters_throws_ExceptionObject(fixed);
    WithDimension<3>::Expect_default_AdvancedBSplineTransform_GetParameters_throws_ExceptionObject(fixed);
  }
}


GTEST_TEST(TransformIO, DefaultElastixFixedParametersAreZeroOrEmpty)
{
  using namespace elastix;

  // Note: This test would fail for AdvancedBSplineTransform, which is related to the test
  // DefaultAdvancedBSplineTransformGetParametersThrowsExceptionObject.
  Expect_default_elastix_FixedParameters_are_all_zero<AdvancedAffineTransformElastix>();
  Expect_default_elastix_FixedParameters_are_all_zero<EulerTransformElastix>();
  Expect_default_elastix_FixedParameters_are_all_zero<SimilarityTransformElastix>();

  Expect_default_elastix_FixedParameters_empty<TranslationTransformElastix>();
}


GTEST_TEST(TransformIO, DefaultElastixParameters)
{
  using namespace elx;

  // Note: This test would fail for AdvancedBSplineTransform, which is related to the test
  // DefaultAdvancedBSplineTransformGetParametersThrowsExceptionObject.

  WithDimension<2>::Expect_default_elastix_Parameters_equal<AdvancedAffineTransformElastix>(
    ExpectedParameters<6>{ 1, 0, 0, 1, 0, 0 });
  WithDimension<3>::Expect_default_elastix_Parameters_equal<AdvancedAffineTransformElastix>(
    ExpectedParameters<12>{ 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0 });
  WithDimension<4>::Expect_default_elastix_Parameters_equal<AdvancedAffineTransformElastix>(
    ExpectedParameters<20>{ 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0 });

  WithDimension<2>::Expect_default_elastix_Parameters_equal<EulerTransformElastix>(ExpectedParameters<3>{});
  WithDimension<3>::Expect_default_elastix_Parameters_equal<EulerTransformElastix>(ExpectedParameters<6>{});
  WithDimension<4>::Expect_default_elastix_Parameters_equal<EulerTransformElastix>(
    ExpectedParameters<20>{ 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0 });

  WithDimension<2>::Expect_default_elastix_Parameters_equal<SimilarityTransformElastix>(
    ExpectedParameters<4>{ 1, 0, 0, 0 });
  WithDimension<3>::Expect_default_elastix_Parameters_equal<SimilarityTransformElastix>(
    ExpectedParameters<7>{ 0, 0, 0, 0, 0, 0, 1 });
  WithDimension<4>::Expect_default_elastix_Parameters_equal<SimilarityTransformElastix>(
    ExpectedParameters<20>{ 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0 });

  WithDimension<2>::Expect_default_elastix_Parameters_equal<TranslationTransformElastix>(ExpectedParameters<2>{});
  WithDimension<3>::Expect_default_elastix_Parameters_equal<TranslationTransformElastix>(ExpectedParameters<3>{});
  WithDimension<4>::Expect_default_elastix_Parameters_equal<TranslationTransformElastix>(ExpectedParameters<4>{});
}


GTEST_TEST(TransformIO, DefaultElastixParametersRemainTheSameWhenSet)
{
  for (const bool fixed : { false, true })
  {
    using namespace elx;

    // Note: This test would fail for AdvancedBSplineTransform, which is related to the test
    // DefaultAdvancedBSplineTransformGetParametersThrowsExceptionObject.

    Expect_default_elastix_Parameters_remain_the_same_when_set<AdvancedAffineTransformElastix>(fixed);
    Expect_default_elastix_Parameters_remain_the_same_when_set<EulerTransformElastix>(fixed);
    Expect_default_elastix_Parameters_remain_the_same_when_set<SimilarityTransformElastix>(fixed);
    Expect_default_elastix_Parameters_remain_the_same_when_set<TranslationTransformElastix>(fixed);
  }
}


GTEST_TEST(TransformIO, CopyDefaultParametersToCorrespondingItkTransform)
{
  for (const bool fixed : { false, true })
  {
    // Note: This test would fail for elx::AdvancedBSplineTransform, which is related to the test
    // DefaultAdvancedBSplineTransformGetParametersThrowsExceptionObject.

    WithDimension<2>::WithElastixTransform<elx::AdvancedAffineTransformElastix>::Test_copying_default_parameters<
      itk::AffineTransform<double, 2>>(fixed);
    WithDimension<3>::WithElastixTransform<elx::AdvancedAffineTransformElastix>::Test_copying_default_parameters<
      itk::AffineTransform<double, 3>>(fixed);
    WithDimension<4>::WithElastixTransform<elx::AdvancedAffineTransformElastix>::Test_copying_default_parameters<
      itk::AffineTransform<double, 4>>(fixed);

    WithDimension<2>::WithElastixTransform<elx::TranslationTransformElastix>::Test_copying_default_parameters<
      itk::TranslationTransform<double, 2>>(fixed);
    WithDimension<3>::WithElastixTransform<elx::TranslationTransformElastix>::Test_copying_default_parameters<
      itk::TranslationTransform<double, 3>>(fixed);
    WithDimension<4>::WithElastixTransform<elx::TranslationTransformElastix>::Test_copying_default_parameters<
      itk::TranslationTransform<double, 4>>(fixed);

    WithDimension<2>::WithElastixTransform<elx::SimilarityTransformElastix>::Test_copying_default_parameters<
      itk::Similarity2DTransform<double>>(fixed);
    WithDimension<3>::WithElastixTransform<elx::SimilarityTransformElastix>::Test_copying_default_parameters<
      itk::Similarity3DTransform<double>>(fixed);

    WithDimension<2>::WithElastixTransform<elx::SimilarityTransformElastix>::Test_copying_default_parameters<
      itk::Euler2DTransform<double>>(fixed);
  }
  WithDimension<3>::WithElastixTransform<elx::EulerTransformElastix>::Test_copying_default_parameters<
    itk::Euler3DTransform<double>>(false);
  // See also CopyDefaultEulerTransformElastix3DFixedParametersToCorrespondingItkTransform
}


GTEST_TEST(TransformIO, CopyDefaultEulerTransformElastix3DFixedParametersToCorrespondingItkTransform)
{
  const auto elxTransform = elx::EulerTransformElastix<ElastixType<3>>::New();
  const auto itkTransform = elx::TransformIO::CreateCorrespondingItkTransform(*elxTransform);
  ASSERT_NE(itkTransform, nullptr);

  const auto elxFixedParameters = elxTransform->GetFixedParameters();
  itkTransform->SetFixedParameters(elxFixedParameters);
  const auto itkFixedParameters = itkTransform->GetFixedParameters();

  // Note: ideally itkFixedParameters and elxFixedParameters should be equal!
  EXPECT_NE(itkFixedParameters, elxFixedParameters);

  ASSERT_GE(itkFixedParameters.size(), elxFixedParameters.size());
  ASSERT_EQ(itkFixedParameters, vnl_vector<double>(4, 0));
}


GTEST_TEST(TransformIO, CopyParametersToCorrespondingItkTransform)
{
  WithDimension<2>::WithElastixTransform<elx::AdvancedAffineTransformElastix>::Test_copying_parameters<
    itk::AffineTransform<double, 2>>();
  WithDimension<3>::WithElastixTransform<elx::AdvancedAffineTransformElastix>::Test_copying_parameters<
    itk::AffineTransform<double, 3>>();

  WithDimension<2>::WithElastixTransform<elx::TranslationTransformElastix>::Test_copying_parameters<
    itk::TranslationTransform<double, 2>>();
  WithDimension<3>::WithElastixTransform<elx::TranslationTransformElastix>::Test_copying_parameters<
    itk::TranslationTransform<double, 3>>();

  WithDimension<2>::WithElastixTransform<elx::SimilarityTransformElastix>::Test_copying_parameters<
    itk::Similarity2DTransform<double>>();
  WithDimension<3>::WithElastixTransform<elx::SimilarityTransformElastix>::Test_copying_parameters<
    itk::Similarity3DTransform<double>>();

  WithDimension<2>::WithElastixTransform<elx::EulerTransformElastix>::Test_copying_parameters<
    itk::Euler2DTransform<double>>();
}


GTEST_TEST(Transform, CreateTransformParametersMapForDefaultTransform)
{
  WithDimension<2>::Test_CreateTransformParametersMap_for_default_transform();
  WithDimension<3>::Test_CreateTransformParametersMap_for_default_transform();
}


GTEST_TEST(Transform, CreateTransformParametersMapDoublePrecision)
{
  // Checks two different transform types, just to be sure.
  WithDimension<2>::WithElastixTransform<
    elx::AdvancedAffineTransformElastix>::Test_CreateTransformParametersMap_double_precision();
  WithDimension<3>::WithElastixTransform<
    elx::TranslationTransformElastix>::Test_CreateTransformParametersMap_double_precision();
}


GTEST_TEST(Transform, CreateTransformParametersSetUseAddition)
{
  // Checks two different transform types, just to be sure.
  WithDimension<2>::WithElastixTransform<
    elx::AdvancedAffineTransformElastix>::Test_CreateTransformParametersMap_SetUseAddition();
  WithDimension<3>::WithElastixTransform<
    elx::TranslationTransformElastix>::Test_CreateTransformParametersMap_SetUseAddition();
}
