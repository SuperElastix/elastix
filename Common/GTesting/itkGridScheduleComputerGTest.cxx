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
#include "itkGridScheduleComputer.h"
#include "elxDefaultConstruct.h"
#include <gtest/gtest.h>


// Checks that the default schedule has consecutive powers of two, in reverse order, for example, for 3D and four
// levels: (GridSpacingSchedule 8 8 8 4 4 4 2 2 2 1 1 1)
GTEST_TEST(GridScheduleComputer, SetDefaultSchedule)
{
  static constexpr auto imageDimension = 3U;

  using GridScheduleComputerType = itk::GridScheduleComputer<double, imageDimension>;
  elx::DefaultConstruct<GridScheduleComputerType> gridScheduleComputer{};

  // In ElastixModelZoo, "NumberOfResolutions" (= number of levels) always appears below 10, so that should be
  // sufficient for this test.
  for (unsigned int numberOfLevels{}; numberOfLevels < 10; ++numberOfLevels)
  {
    gridScheduleComputer.SetDefaultSchedule(numberOfLevels);
    GridScheduleComputerType::VectorGridSpacingFactorType schedule{};
    gridScheduleComputer.GetSchedule(schedule);
    EXPECT_EQ(schedule.size(), numberOfLevels);

    for (unsigned int level = 0; level < numberOfLevels; ++level)
    {
      for (const double actualScheduleValue : schedule.at(level))
      {
        EXPECT_DOUBLE_EQ(actualScheduleValue, std::pow(2.0, numberOfLevels - level - 1));
      }
    }
  }
}
