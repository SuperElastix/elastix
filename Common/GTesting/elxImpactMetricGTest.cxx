#include "Impact/elxImpactMetric.h"
#include <gtest/gtest.h>
#include <limits>
#include <string>
#include <vector>
#include <sstream>
#include "itkParameterMapInterface.h"

using elx::GroupByDimensions;

namespace
{
// Helper to format vector outputs
template <typename T>
std::string
vecToStr(const std::vector<T> & vec)
{
  std::ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < vec.size(); ++i)
    oss << vec[i] << (i + 1 < vec.size() ? ", " : "");
  oss << "]";
  return oss.str();
}

template <typename T>
void
ExpectVectorEqual(const std::vector<T> & actual, const std::vector<T> & expected)
{
  EXPECT_EQ(actual, expected) << "Expected: " << vecToStr(expected) << "\nActual:   " << vecToStr(actual);
}
} // namespace

GTEST_TEST(GroupByDimensions, Basic)
{
  std::vector<unsigned int>              dimensions = { 3, 3 };
  std::vector<unsigned int>              values1 = { 5, 5, 5, 8, 6, 7 };
  std::vector<std::vector<unsigned int>> resultVec1 = GroupByDimensions<unsigned int>(values1, dimensions);

  EXPECT_EQ(resultVec1.size(), 2);
  ExpectVectorEqual(resultVec1[0], { 5, 5, 5 });
  ExpectVectorEqual(resultVec1[1], { 8, 6, 7 });

  std::vector<float>              values2 = { 5, 5, 5, 8, 6 };
  std::vector<std::vector<float>> resultVec2 = GroupByDimensions<float>(values2, dimensions);

  EXPECT_EQ(resultVec2.size(), 2);
  ExpectVectorEqual(resultVec2[0], { 5, 5, 5 });
  ExpectVectorEqual(resultVec2[1], { 8, 6, 8 });

  std::vector<float>              values3 = { 5, 5, 5, 8, 6, 7, 8 };
  std::vector<std::vector<float>> resultVec3 = GroupByDimensions<float>(values3, dimensions);

  EXPECT_EQ(resultVec3.size(), 2);
  ExpectVectorEqual(resultVec3[0], { 5, 5, 5 });
  ExpectVectorEqual(resultVec3[1], { 8, 6, 7 });
}
