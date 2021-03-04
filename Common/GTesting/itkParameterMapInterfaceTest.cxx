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
#include "itkParameterMapInterface.h"

#include <itkNumberToString.h>

#include <gtest/gtest.h>


// The class to be tested.
using itk::ParameterMapInterface;


GTEST_TEST(ParameterMapInterface, RetrieveValuesReturnsNullWhenParameterMapIsEmpty)
{
  const auto parameterMapInterface = ParameterMapInterface::New();
  EXPECT_EQ(parameterMapInterface->RetrieveValues<double>("parameterName"), nullptr);
}


GTEST_TEST(ParameterMapInterface, RetrieveValuesReturnsNullWhenParameterNameIsMissing)
{
  const auto parameterMapInterface = ParameterMapInterface::New();
  parameterMapInterface->SetParameterMap({ { "ParameterName", { "0", "1" } } });

  EXPECT_EQ(parameterMapInterface->RetrieveValues<double>("MissingParameterName"), nullptr);
}


GTEST_TEST(ParameterMapInterface, RetrieveValuesSupportsZeroValues)
{
  const auto        parameterMapInterface = ParameterMapInterface::New();
  const std::string parameterName("Key");
  parameterMapInterface->SetParameterMap({ { parameterName, {} } });

  const auto retrievedValues = parameterMapInterface->RetrieveValues<double>(parameterName);
  ASSERT_NE(retrievedValues, nullptr);
  EXPECT_EQ(*retrievedValues, std::vector<double>{});
}


GTEST_TEST(ParameterMapInterface, RetrieveValuesSupportsSingleValue)
{
  const auto        parameterMapInterface = ParameterMapInterface::New();
  const std::string parameterName("Key");

  for (const double testValue : { -1.0, 0.0, 1.0 })
  {
    parameterMapInterface->SetParameterMap({ { parameterName, { itk::NumberToString<double>{}(testValue) } } });

    const auto retrievedValues = parameterMapInterface->RetrieveValues<double>(parameterName);
    ASSERT_NE(retrievedValues, nullptr);
    EXPECT_EQ(*retrievedValues, std::vector<double>{ testValue });
  }
}


GTEST_TEST(ParameterMapInterface, RetrieveValuesSupportsMultipleValues)
{
  const auto        parameterMapInterface = ParameterMapInterface::New();
  const std::string parameterName("Key");
  using NumberToString = itk::NumberToString<double>;

  for (const double testValue1 : { 0.0, 1.0 })
  {
    for (const double testValue2 : { 0.0, 1.0 })
    {
      parameterMapInterface->SetParameterMap(
        { { parameterName, { NumberToString{}(testValue1), NumberToString{}(testValue2) } } });

      const auto retrievedValues = parameterMapInterface->RetrieveValues<double>(parameterName);
      ASSERT_NE(retrievedValues, nullptr);
      const std::vector<double> expectedValues{ testValue1, testValue2 };
      EXPECT_EQ(*retrievedValues, expectedValues);
    }
  }
}


GTEST_TEST(ParameterMapInterface, RetrieveValuesThrowsExceptionWhenConversionFails)
{
  const auto        parameterMapInterface = ParameterMapInterface::New();
  const std::string parameterName("Key");
  parameterMapInterface->SetParameterMap({ { parameterName, { "not-a-valid-floating-point-value" } } });

  EXPECT_THROW(parameterMapInterface->RetrieveValues<double>(parameterName), itk::ExceptionObject);
}
