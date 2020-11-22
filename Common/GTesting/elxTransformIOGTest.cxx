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

#include "elxElastixTemplate.h"
#include "AdvancedAffineTransform/elxAdvancedAffineTransform.h"
#include "AdvancedBSplineTransform/elxAdvancedBSplineTransform.h"
#include "EulerTransform/elxEulerTransform.h"
#include "SimilarityTransform/elxSimilarityTransform.h"
#include "TranslationTransform/elxTranslationTransform.h"

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
