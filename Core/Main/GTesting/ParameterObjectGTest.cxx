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
