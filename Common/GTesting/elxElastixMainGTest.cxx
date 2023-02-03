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
#include "elxElastixMain.h"

#include <gtest/gtest.h>


// Tests retrieving the component data base and a component creator in parallel.
GTEST_TEST(ElastixMain, GetComponentDatabaseAndCreatorInParallel)
{
#pragma omp parallel for
  for (auto i = 0; i <= 9; ++i)
  {
    const auto creator = elx::ElastixMain::GetComponentDatabase().GetCreator("Elastix", 1);
    EXPECT_NE(creator, nullptr);
    EXPECT_NE(creator, elx::ComponentDatabase::PtrToCreator{});

    if (creator != nullptr)
    {
      const auto elxComponent = creator();
      EXPECT_NE(elxComponent, nullptr);
      EXPECT_NE(elxComponent, itk::Object::Pointer{});
      EXPECT_NE(dynamic_cast<elx::ElastixBase *>(elxComponent.GetPointer()), nullptr);
    }
  }
}
