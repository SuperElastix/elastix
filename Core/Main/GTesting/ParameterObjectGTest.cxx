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
#include "elxParameterObject.h"

#include "GTesting/elxGTestUtilities.h"

#include "elxCoreMainGTestUtilities.h"
#include "elxDefaultConstruct.h"

// ITK header file:
#include <itkFileTools.h>
#include <itksys/SystemTools.hxx>

// GoogleTest header file:
#include <gtest/gtest.h>

#include <string>

// Type aliases:
using ParameterMapType = elx::ParameterObject::ParameterMapType;
using ParameterMapVectorType = elx::ParameterObject::ParameterMapVectorType;
using ParameterFileNameVectorType = elx::ParameterObject::ParameterFileNameVectorType;

// Using-declarations:
using elx::CoreMainGTestUtilities::GetCurrentBinaryDirectoryPath;
using elx::CoreMainGTestUtilities::GetNameOfTest;
using elx::CoreMainGTestUtilities::TypeHolder;

// Tests that ParameterObject::WriteParameterFiles writes all the specified files.
GTEST_TEST(ParameterObject, WriteParameterFiles)
{
  const std::string rootOutputDirectoryPath = GetCurrentBinaryDirectoryPath() + '/' + GetNameOfTest(*this);
  itk::FileTools::CreateDirectory(rootOutputDirectoryPath);

  elx::DefaultConstruct<elx::ParameterObject> parameterObject{};

  for (const std::size_t numberOfMaps : { 0, 1, 2 })
  {
    const std::string outputDirectoryPath = rootOutputDirectoryPath + '/' + std::to_string(numberOfMaps);
    itk::FileTools::CreateDirectory(outputDirectoryPath);

    parameterObject.SetParameterMaps(ParameterMapVectorType(numberOfMaps));

    ParameterFileNameVectorType fileNames{};

    for (std::size_t i{}; i < numberOfMaps; ++i)
    {
      fileNames.push_back(outputDirectoryPath + '/' + "ParameterFile." + std::to_string(i) + ".txt");
    }

    parameterObject.WriteParameterFiles(fileNames);

    // Check that each of the specified files is written to disk.
    for (const auto & fileName : fileNames)
    {
      EXPECT_TRUE(itksys::SystemTools::FileExists(fileName.c_str(), true));
    }
  }
}


// Tests that any numeric parameter value is printed (by `<<` into an output stream) literally as it was originally
// placed into the ParameterMap.
GTEST_TEST(ParameterObject, PrintOriginalNumericValues)
{
  static constexpr auto check = [](const std::string & parameterValue) {
    elx::DefaultConstruct<elx::ParameterObject> parameterObject{};

    const std::string parameterName = "ParameterName";

    parameterObject.SetParameterMap(elx::ParameterObject::ParameterMapType{ { parameterName, { parameterValue } } });
    std::ostringstream outputStringStream;

    outputStringStream << parameterObject;
    const auto        printedString = outputStringStream.str();
    const std::string expectedParameter = '(' + parameterName + ' ' + parameterValue + ")\n";
    ASSERT_GT(printedString.size(), expectedParameter.size());
    EXPECT_EQ(std::string_view(printedString).substr(printedString.size() - expectedParameter.size()),
              expectedParameter);
  };

  for (const std::string parameterValue : { "0", "1", "1.23456789", "0.1", "-0.1", "3e+38", "4e+38" })
  {
    check(parameterValue);
  }

  check(std::to_string(std::numeric_limits<std::int32_t>::min()));
  check(std::to_string(std::numeric_limits<std::int32_t>::max()));
  check(std::to_string(std::numeric_limits<std::uint32_t>::max()));
  check(std::to_string(std::numeric_limits<std::int64_t>::min()));
  check(std::to_string(std::numeric_limits<std::int64_t>::max()));
  check(std::to_string(std::numeric_limits<std::uint64_t>::max()));

  {
    // Test +/- 100000000000000000000
    constexpr auto power_of_ten = 10.0 * static_cast<double>(itk::Math::UnsignedPower(10, 19));
    check(elx::Conversion::ToString(power_of_ten));
    check(elx::Conversion::ToString(-power_of_ten));
  }
  {
    // Test +/- 999999999999999
    constexpr auto power_of_ten = static_cast<double>(itk::Math::UnsignedPower(10, 15));
    check(elx::Conversion::ToString(power_of_ten - 1.0));
    check(elx::Conversion::ToString(1.0 - power_of_ten));
  }
  {
    // Test +/- 0.000001
    constexpr auto power_of_ten = 1.0 / static_cast<double>(itk::Math::UnsignedPower(10, 6));
    check(elx::Conversion::ToString(power_of_ten));
    check(elx::Conversion::ToString(-power_of_ten));
  }

  const auto checkNumericLimits = [](const auto typeHolder) {
    (void)typeHolder;
    using FloatingPointType = typename decltype(typeHolder)::Type;
    using NumericLimits = std::numeric_limits<FloatingPointType>;
    for (const auto value : { NumericLimits::epsilon(),
                              NumericLimits::min(),
                              NumericLimits::lowest(),
                              NumericLimits::max(),
                              NumericLimits::denorm_min(),
                              -NumericLimits::denorm_min() })
    {
      check(elx::Conversion::ToString(value));
    }
  };

  checkNumericLimits(TypeHolder<float>{});
  checkNumericLimits(TypeHolder<double>{});
}
