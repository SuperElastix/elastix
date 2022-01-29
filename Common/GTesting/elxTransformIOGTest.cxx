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

#define _USE_MATH_DEFINES // For M_PI_4.

// First include the header file to be tested:
#include "elxTransformIO.h"

#include "elxConversion.h"
#include "elxDefaultConstructibleSubclass.h"
#include "elxElastixMain.h" // For xoutManager.
#include "elxElastixTemplate.h"
#include "elxGTestUtilities.h"
#include "../Core/Main/GTesting/elxCoreMainGTestUtilities.h"

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

#include <itkAffineTransform.h>
#include <itkBSplineTransform.h>
#include <itkCompositeTransform.h>
#include <itkEuler2DTransform.h>
#include <itkEuler3DTransform.h>
#include <itkSimilarity2DTransform.h>
#include <itkSimilarity3DTransform.h>
#include <itkTranslationTransform.h>

#include <itkImage.h>
#include <itkIndexRange.h>
#include <itkVector.h>

#include <itksys/SystemTools.hxx>

#include <cmath> // For M_PI_4.
#include <typeinfo>
#include <type_traits> // For extent, is_base_of and is_same

#include <gtest/gtest.h>

using ParameterValuesType = itk::ParameterFileParser::ParameterValuesType;
using ParameterMapType = itk::ParameterFileParser::ParameterMapType;


// Using-declarations:
using elx::CoreMainGTestUtilities::CheckNew;
using elx::GTestUtilities::CreateDefaultElastixObject;
using elx::GTestUtilities::ExpectAllKeysUnique;
using elx::GTestUtilities::GeneratePseudoRandomParameters;
using elx::GTestUtilities::MakeMergedMap;
using elx::GTestUtilities::MakePoint;
using elx::GTestUtilities::MakeVector;


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
  template <template <typename> class TElastixTransform>
  struct WithElastixTransform
  {
    using ElastixTransformType = TElastixTransform<ElastixType<NDimension>>;

    template <typename TExpectedCorrespondingItkTransform>
    static void
    Expect_CorrespondingItkCompositeTransform()
    {
      using ItkCompositeTransformType = itk::CompositeTransform<double, NDimension>;

      static_assert(
        std::is_base_of<itk::Transform<double, NDimension, NDimension>, TExpectedCorrespondingItkTransform>::value,
        "TExpectedCorrespondingItkTransform should be derived from the expected `itk::Transform` specialization!");
      static_assert(!std::is_base_of<ItkCompositeTransformType, TExpectedCorrespondingItkTransform>::value,
                    "TExpectedCorrespondingItkTransform should not be derived from `itk::CompositeTransform`!");

      const auto elxTransform = CheckNew<ElastixTransformType>();

      EXPECT_EQ(elxTransform->elxGetClassName(),
                elx::TransformIO::ConvertITKNameOfClassToElastixClassName(
                  elx::DefaultConstructibleSubclass<TExpectedCorrespondingItkTransform>{}.GetNameOfClass()));

      const auto compositeTransform = elx::TransformIO::ConvertToItkCompositeTransform(*elxTransform);
      ASSERT_NE(compositeTransform, nullptr);
      static_assert(
        std::is_same<const itk::SmartPointer<ItkCompositeTransformType>, decltype(compositeTransform)>::value,
        "`ConvertToItkCompositeTransform` should have the expected `SmartPointer` return type!");

      const auto & transformQueue = compositeTransform->GetTransformQueue();
      ASSERT_EQ(transformQueue.size(), 1);
      const auto & itkTransform = transformQueue.front();
      ASSERT_NE(itkTransform, nullptr);

      const auto & actualItkTransformTypeId = typeid(*itkTransform);
      const auto & expectedItkTransformTypeId = typeid(TExpectedCorrespondingItkTransform);
      ASSERT_EQ(std::string(actualItkTransformTypeId.name()), std::string(expectedItkTransformTypeId.name()));
      EXPECT_EQ(actualItkTransformTypeId, expectedItkTransformTypeId);
    }


    static void
    Expect_default_elastix_FixedParameters_empty()
    {
      const auto fixedParameters = CheckNew<ElastixTransformType>()->GetFixedParameters();
      ASSERT_EQ(fixedParameters, vnl_vector<double>());
    }


    static void
    Expect_default_elastix_FixedParameters_are_all_zero()
    {
      const auto fixedParameters = CheckNew<ElastixTransformType>()->GetFixedParameters();
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
                     .append(elx::Conversion::BoolToString(fixed)));

      const auto transform = CheckNew<ElastixTransformType>();
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
                     .append(elx::Conversion::BoolToString(fixed)));

      const auto elxTransform = CheckNew<ElastixTransformType>();
      SCOPED_TRACE(fixed);

      const auto compositeTransform = elx::TransformIO::ConvertToItkCompositeTransform(*elxTransform);
      ASSERT_NE(compositeTransform, nullptr);
      const auto & transformQueue = compositeTransform->GetTransformQueue();
      ASSERT_EQ(transformQueue.size(), 1);
      const auto & itkTransform = transformQueue.front();
      ASSERT_NE(itkTransform, nullptr);

      const auto parameters = elx::TransformIO::GetParameters(fixed, *elxTransform);
      elx::TransformIO::SetParameters(fixed, *itkTransform, parameters);

      ASSERT_EQ(elx::TransformIO::GetParameters(fixed, *itkTransform), parameters);
    }

    template <typename TExpectedCorrespondingItkTransform>
    static void
    Test_copying_parameters()
    {
      const auto elxTransform = CheckNew<ElastixTransformType>();

      SCOPED_TRACE(elxTransform->elxGetClassName());

      const auto compositeTransform = elx::TransformIO::ConvertToItkCompositeTransform(*elxTransform);
      ASSERT_NE(compositeTransform, nullptr);
      const auto & transformQueue = compositeTransform->GetTransformQueue();
      ASSERT_EQ(transformQueue.size(), 1);
      const auto & itkTransform = transformQueue.front();
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
    Test_CreateTransformParametersMap_for_default_transform(const ParameterMapType & expectedDerivedParameterMap)
    {
      SCOPED_TRACE(std::string("Function = ")
                     .append(__func__)
                     .append("\n  ElastixTransformType = ")
                     .append(typeid(ElastixTransformType).name()));

      const elx::xoutManager manager("", false, false);

      const auto elxTransform = CheckNew<ElastixTransformType>();
      const auto elastixObject = CreateDefaultElastixObject<ElastixType<NDimension>>();

      // Note: SetElastix does not take or share the ownership of its argument!
      elxTransform->SetElastix(elastixObject);

      // BeforeAll() appears necessary to for MultiBSplineTransformWithNormal
      // and AdvancedBSplineTransform to initialize the internal ITK
      // transform of the elastix transform, by calling
      // InitializeBSplineTransform()
      elxTransform->BeforeAll();

      // Overrule the default-constructors of BSplineTransformWithDiffusion
      // and DeformationFieldTransform which do SetReadWriteTransformParameters(false)
      elxTransform->SetReadWriteTransformParameters(true);

      ParameterMapType actualParameterMap;
      elxTransform->CreateTransformParametersMap(itk::OptimizerParameters<double>{}, actualParameterMap);

      const std::string expectedImageDimension{ char{ '0' + NDimension } };
      const std::string expectedInternalImagePixelType = "float";
      const std::string expectedZero = "0";
      const std::string expectedOne = "1";

      const ParameterMapType expectedBaseParameterMap = {
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

      ExpectAllKeysUnique(expectedDerivedParameterMap, expectedBaseParameterMap);
      EXPECT_EQ(actualParameterMap, MakeMergedMap(expectedDerivedParameterMap, expectedBaseParameterMap));
    }

    static void
    Test_CreateTransformParametersMap_double_precision()
    {
      const elx::xoutManager manager("", false, false);

      // Use 0.3333333333333333... as test value.
      constexpr auto testValue = 1.0 / 3.0;
      constexpr auto expectedPrecision = 16;
      static_assert(expectedPrecision == std::numeric_limits<double>::digits10 + 1,
                    "The expected precision for double floating point numbers");
      const auto expectedString = "0." + std::string(expectedPrecision, '3');

      const auto elastixObject = CheckNew<ElastixType<NDimension>>();
      elastixObject->SetConfiguration(CheckNew<elx::Configuration>());

      const auto imageContainer = elx::ElastixBase::DataObjectContainerType::New();
      const auto image = itk::Image<float, NDimension>::New();
      image->SetOrigin(testValue);
      image->SetSpacing(testValue);
      imageContainer->push_back(image);

      elastixObject->SetFixedImageContainer(imageContainer);
      elastixObject->SetMovingImageContainer(imageContainer);

      const auto elxTransform = CheckNew<ElastixTransformType>();
      elxTransform->SetElastix(elastixObject);
      elxTransform->BeforeAll();

      ParameterMapType parameterMap;

      elxTransform->CreateTransformParametersMap(itk::OptimizerParameters<double>(2U, testValue), parameterMap);

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

      const auto elxTransform = CheckNew<ElastixTransformType>();

      const auto elastixObject = CreateDefaultElastixObject<ElastixType<NDimension>>();

      // Note: SetElastix does not take or share the ownership of its argument!
      elxTransform->SetElastix(elastixObject);

      elxTransform->BeforeAll();

      const auto expectHowToCombineTransforms = [&elxTransform](const char * const expectedParameterValue) {
        ParameterMapType parameterMap;
        elxTransform->CreateTransformParametersMap({}, parameterMap);

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
    const auto transform = CheckNew<elx::AdvancedBSplineTransform<ElastixType<NDimension>>>();
    EXPECT_THROW(elx::TransformIO::GetParameters(fixed, *transform), itk::ExceptionObject);
  }


  template <template <typename> class TElastixTransform, std::size_t NExpectedParameters>
  static void
  Expect_default_elastix_Parameters_equal(const ExpectedParameters<NExpectedParameters> & expectedParameters)
  {
    using ElastixTransformType = TElastixTransform<ElastixType<NDimension>>;

    SCOPED_TRACE(std::string("Function = ")
                   .append(__func__)
                   .append("\n  ElastixTransformType = ")
                   .append(typeid(ElastixTransformType).name()));

    const auto parameters = CheckNew<ElastixTransformType>()->GetParameters();
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
    const std::string expectedDeformationFieldFileName("DeformationFieldImage.mhd");

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
      { { "DeformationFieldFileName", { expectedDeformationFieldFileName } },
        { "GridIndex", expectedZeros },
        { "GridOrigin", expectedZeros },
        { "GridSize", expectedZeros },
        { "GridSpacing", expectedOnes } });
    WithElastixTransform<DeformationFieldTransform>::Test_CreateTransformParametersMap_for_default_transform(
      { { "DeformationFieldFileName", { expectedDeformationFieldFileName } },
        { "DeformationFieldInterpolationOrder", { expectedZero } } });
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


template <template <typename> class TElastixTransform>
void
Expect_default_elastix_FixedParameters_are_all_zero()
{
  WithDimension<2>::WithElastixTransform<TElastixTransform>::Expect_default_elastix_FixedParameters_are_all_zero();
  WithDimension<3>::WithElastixTransform<TElastixTransform>::Expect_default_elastix_FixedParameters_are_all_zero();
  WithDimension<4>::WithElastixTransform<TElastixTransform>::Expect_default_elastix_FixedParameters_are_all_zero();
}


template <template <typename> class TElastixTransform>
void
Expect_default_elastix_FixedParameters_empty()
{
  WithDimension<2>::WithElastixTransform<TElastixTransform>::Expect_default_elastix_FixedParameters_empty();
  WithDimension<3>::WithElastixTransform<TElastixTransform>::Expect_default_elastix_FixedParameters_empty();
  WithDimension<4>::WithElastixTransform<TElastixTransform>::Expect_default_elastix_FixedParameters_empty();
}


template <template <typename> class TElastixTransform>
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


template <template <typename> class TElastixTransform, typename TITKTransform>
void
Expect_elx_TransformPoint_yields_same_point_as_ITK(const TITKTransform & itkTransform)
{
  const auto Dimension = TITKTransform::SpaceDimension;

  const auto elastixObject = CreateDefaultElastixObject<ElastixType<Dimension>>();

  const auto elxTransform = CheckNew<TElastixTransform<ElastixType<Dimension>>>();

  const std::string elxClassName = elxTransform->elxGetClassName();
  const std::string itkNameOfClass = itkTransform.GetNameOfClass();

  if (!((elxClassName == "BSplineTransform") && (itkNameOfClass == elxClassName)))
  {
    // Check that the elastix transform type corresponds with the ITK transform type.
    EXPECT_EQ(elxClassName, elx::TransformIO::ConvertITKNameOfClassToElastixClassName(itkNameOfClass));
  }

  // Note: SetElastix does not take or share the ownership of its argument!
  elxTransform->SetElastix(elastixObject);

  // Necessary for AdvancedBSplineTransform, to avoid an exception, saying
  // "No current transform set in the AdvancedCombinationTransform".
  elxTransform->BeforeAll();

  // SetFixedParameters before SetParameters, to avoid an exception from
  // AdvancedBSplineTransform, saying "AdvancedBSplineDeformableTransform:
  // Mismatched between parameters size 32 and region size 0"
  elxTransform->SetFixedParameters(itkTransform.GetFixedParameters());
  elxTransform->SetParameters(itkTransform.GetParameters());

  unsigned numberOfTimesOutputDiffersFromInput{};
  unsigned numberOfTimesOutputIsNonZero{};

  using NumericLimits = std::numeric_limits<double>;

  constexpr double testInputValues[] = { NumericLimits::lowest(),
                                         -2.0,
                                         -1.0,
                                         -0.5,
                                         -NumericLimits::min(),
                                         -0.0,
                                         0.0,
                                         NumericLimits::min(),
                                         0.5,
                                         1.0 - (2 * NumericLimits::epsilon()), // Note: 1.0 fails on BSpline!!!
                                         1.0 + 1.0e-14,
                                         2.0,
                                         NumericLimits::max() };

  constexpr auto numberOfTestInputValues = std::extent<decltype(testInputValues)>::value;

  // Use the test input values as coordinates.
  for (const auto index : itk::ZeroBasedIndexRange<Dimension>(itk::Size<Dimension>::Filled(numberOfTestInputValues)))
  {
    itk::Point<double, Dimension> inputPoint;
    std::transform(index.begin(), index.end(), inputPoint.begin(), [&testInputValues](const itk::SizeValueType value) {
      return testInputValues[value];
    });

    const auto expectedOutputPoint = itkTransform.TransformPoint(inputPoint);
    const auto actualOutputPoint = elxTransform->TransformPoint(inputPoint);

    static_assert(std::is_same<decltype(actualOutputPoint), decltype(expectedOutputPoint)>::value,
                  "elxTransform->TransformPoint must have the expected return type!");

    if (expectedOutputPoint != inputPoint)
    {
      ++numberOfTimesOutputDiffersFromInput;
    }
    if (expectedOutputPoint != itk::Point<double, Dimension>())
    {
      ++numberOfTimesOutputIsNonZero;
    }

    const auto pointToString = [](const itk::Point<double, Dimension> & point) {
      std::ostringstream stream;
      stream << point;
      return stream.str();
    };

    if (pointToString(actualOutputPoint) != pointToString(expectedOutputPoint))
    {
      // Consider two points significantly different when they have different string representations.
      EXPECT_EQ(actualOutputPoint, expectedOutputPoint) << " inputPoint = " << inputPoint;
    }
  }

  // If the output point would always equal the input point, either the test
  // or the transform might not make much sense.
  EXPECT_GT(numberOfTimesOutputDiffersFromInput, 0U);

  // If the output point would always be zero (0, 0, 0), again, either the test
  // or the transform might not make much sense.
  EXPECT_GT(numberOfTimesOutputIsNonZero, 0U);
}

} // namespace


// Tests that elx::TransformIO::ConvertToItkCompositeTransform(elxTransform) yields the expected corresponding
// `itk::CompositeTransform`.
GTEST_TEST(TransformIO, CorrespondingItkCompositeTransform)
{
  WithDimension<2>::WithElastixTransform<
    elx::AdvancedAffineTransformElastix>::Expect_CorrespondingItkCompositeTransform<itk::AffineTransform<double, 2>>();
  WithDimension<3>::WithElastixTransform<
    elx::AdvancedAffineTransformElastix>::Expect_CorrespondingItkCompositeTransform<itk::AffineTransform<double, 3>>();
  WithDimension<4>::WithElastixTransform<
    elx::AdvancedAffineTransformElastix>::Expect_CorrespondingItkCompositeTransform<itk::AffineTransform<double, 4>>();

  // Note: This test fails for `elx::AdvancedBSplineTransform` (corresponding with `itk::BSplineTransform`), as it
  // produces an `itk::ExceptionObject`: unknown file: error: C++ exception with description
  // "<source-directory>\elastix\Common\Transforms\itkAdvancedCombinationTransform.hxx:223: ITK ERROR:
  // AdvancedBSplineTransform(0000018BCE262040): No current transform set in the AdvancedCombinationTransform" thrown in
  // the test body.

  WithDimension<2>::WithElastixTransform<elx::TranslationTransformElastix>::Expect_CorrespondingItkCompositeTransform<
    itk::TranslationTransform<double, 2>>();
  WithDimension<3>::WithElastixTransform<elx::TranslationTransformElastix>::Expect_CorrespondingItkCompositeTransform<
    itk::TranslationTransform<double, 3>>();
  WithDimension<4>::WithElastixTransform<elx::TranslationTransformElastix>::Expect_CorrespondingItkCompositeTransform<
    itk::TranslationTransform<double, 4>>();

  WithDimension<2>::WithElastixTransform<elx::SimilarityTransformElastix>::Expect_CorrespondingItkCompositeTransform<
    itk::Similarity2DTransform<double>>();
  WithDimension<3>::WithElastixTransform<elx::SimilarityTransformElastix>::Expect_CorrespondingItkCompositeTransform<
    itk::Similarity3DTransform<double>>();

  WithDimension<2>::WithElastixTransform<elx::EulerTransformElastix>::Expect_CorrespondingItkCompositeTransform<
    itk::Euler2DTransform<double>>();
  WithDimension<3>::WithElastixTransform<elx::EulerTransformElastix>::Expect_CorrespondingItkCompositeTransform<
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
  const auto elxTransform = CheckNew<elx::EulerTransformElastix<ElastixType<3>>>();
  const auto compositeTransform = elx::TransformIO::ConvertToItkCompositeTransform(*elxTransform);
  ASSERT_NE(compositeTransform, nullptr);
  const auto & transformQueue = compositeTransform->GetTransformQueue();
  ASSERT_EQ(transformQueue.size(), 1);
  const auto & itkTransform = transformQueue.front();
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


GTEST_TEST(Transform, TransformedPointSameAsITKTranslation2D)
{
  constexpr auto Dimension = 2U;

  elx::DefaultConstructibleSubclass<itk::TranslationTransform<double, Dimension>> itkTransform;
  itkTransform.SetOffset(MakeVector(1.0, 2.0));

  Expect_elx_TransformPoint_yields_same_point_as_ITK<elx::TranslationTransformElastix>(itkTransform);
}


GTEST_TEST(Transform, TransformedPointSameAsITKTranslation3D)
{
  constexpr auto Dimension = 3U;

  elx::DefaultConstructibleSubclass<itk::TranslationTransform<double, Dimension>> itkTransform;
  itkTransform.SetOffset(MakeVector(1.0, 2.0, 3.0));

  Expect_elx_TransformPoint_yields_same_point_as_ITK<elx::TranslationTransformElastix>(itkTransform);
}


GTEST_TEST(Transform, TransformedPointSameAsITKAffine2D)
{
  constexpr auto Dimension = 2U;

  elx::DefaultConstructibleSubclass<itk::AffineTransform<double, Dimension>> itkTransform;
  itkTransform.SetTranslation(MakeVector(1.0, 2.0));
  itkTransform.Scale(MakeVector(1.5, 1.75));
  itkTransform.SetCenter(MakePoint(0.5, 1.5));
  itkTransform.Rotate2D(M_PI_4);

  Expect_elx_TransformPoint_yields_same_point_as_ITK<elx::AdvancedAffineTransformElastix>(itkTransform);
}


GTEST_TEST(Transform, TransformedPointSameAsITKAffine3D)
{
  constexpr auto Dimension = 3U;

  elx::DefaultConstructibleSubclass<itk::AffineTransform<double, Dimension>> itkTransform;
  itkTransform.SetTranslation(MakeVector(1.0, 2.0, 3.0));
  itkTransform.SetCenter(MakePoint(3.0, 2.0, 1.0));
  itkTransform.Scale(MakeVector(1.25, 1.5, 1.75));
  itkTransform.Rotate3D(itk::Vector<double, Dimension>(1.0), M_PI_4);

  Expect_elx_TransformPoint_yields_same_point_as_ITK<elx::AdvancedAffineTransformElastix>(itkTransform);
}


GTEST_TEST(Transform, TransformedPointSameAsITKEuler2D)
{
  elx::DefaultConstructibleSubclass<itk::Euler2DTransform<double>> itkTransform;
  itkTransform.SetTranslation(MakeVector(1.0, 2.0));
  itkTransform.SetCenter(MakePoint(0.5, 1.5));
  itkTransform.SetAngle(M_PI_4);

  Expect_elx_TransformPoint_yields_same_point_as_ITK<elx::EulerTransformElastix>(itkTransform);
}


GTEST_TEST(Transform, TransformedPointSameAsITKEuler3D)
{
  elx::DefaultConstructibleSubclass<itk::Euler3DTransform<double>> itkTransform;
  itkTransform.SetTranslation(MakeVector(1.0, 2.0, 3.0));
  itkTransform.SetCenter(MakePoint(3.0, 2.0, 1.0));
  itkTransform.SetRotation(M_PI_2, M_PI_4, M_PI_4 / 2.0);

  Expect_elx_TransformPoint_yields_same_point_as_ITK<elx::EulerTransformElastix>(itkTransform);
}


GTEST_TEST(Transform, TransformedPointSameAsITKSimilarity2D)
{
  elx::DefaultConstructibleSubclass<itk::Similarity2DTransform<double>> itkTransform;
  itkTransform.SetScale(0.75);
  itkTransform.SetTranslation(MakeVector(1.0, 2.0));
  itkTransform.SetCenter(MakePoint(0.5, 1.5));
  itkTransform.SetAngle(M_PI_4);

  Expect_elx_TransformPoint_yields_same_point_as_ITK<elx::SimilarityTransformElastix>(itkTransform);
}


GTEST_TEST(Transform, TransformedPointSameAsITKSimilarity3D)
{
  elx::DefaultConstructibleSubclass<itk::Similarity3DTransform<double>> itkTransform;
  itkTransform.SetScale(0.75);
  itkTransform.SetTranslation(MakeVector(1.0, 2.0, 3.0));
  itkTransform.SetCenter(MakePoint(3.0, 2.0, 1.0));
  itkTransform.SetRotation(itk::Vector<double, 3>(1.0), M_PI_4);

  Expect_elx_TransformPoint_yields_same_point_as_ITK<elx::SimilarityTransformElastix>(itkTransform);
}


GTEST_TEST(Transform, TransformedPointSameAsITKBSpline2D)
{
  elx::DefaultConstructibleSubclass<itk::BSplineTransform<double, 2>> itkTransform;
  itkTransform.SetParameters(GeneratePseudoRandomParameters(itkTransform.GetParameters().size(), -1.0));

  Expect_elx_TransformPoint_yields_same_point_as_ITK<elx::AdvancedBSplineTransform>(itkTransform);
  Expect_elx_TransformPoint_yields_same_point_as_ITK<elx::RecursiveBSplineTransform>(itkTransform);
}

GTEST_TEST(Transform, TransformedPointSameAsITKBSpline3D)
{
  elx::DefaultConstructibleSubclass<itk::BSplineTransform<double, 3>> itkTransform;
  itkTransform.SetParameters(GeneratePseudoRandomParameters(itkTransform.GetParameters().size(), -1.0));

  Expect_elx_TransformPoint_yields_same_point_as_ITK<elx::AdvancedBSplineTransform>(itkTransform);
  Expect_elx_TransformPoint_yields_same_point_as_ITK<elx::RecursiveBSplineTransform>(itkTransform);
}
