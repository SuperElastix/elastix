#include "Impact/elxImpactMetric.h"
#include <gtest/gtest.h>
#include <limits>
#include <string>
#include <vector>
#include <sstream>
#include "itkParameterMapInterface.h"

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


GTEST_TEST(GetVectorFromString, LimitsAndParsing)
{
  // ---------- std::string ----------
  ExpectVectorEqual(GetVectorFromString<std::string>("a 2   -1c", "default"), { "a", "2", "-1c" });
  ExpectVectorEqual(GetVectorFromString<std::string>("", "default"), { "default" });
  ExpectVectorEqual(GetVectorFromString<std::string>(5, "a b", "default"), { "a", "b", "a", "a", "a" });
  ExpectVectorEqual(GetVectorFromString<std::string>(3, "", "default"), { "default", "default", "default" });

  // ---------- unsigned int ----------
  ExpectVectorEqual(GetVectorFromString<unsigned int>("6 18", 42), { 6, 18 });
  ExpectVectorEqual(GetVectorFromString<unsigned int>("", 42), { 42 });
  ExpectVectorEqual(GetVectorFromString<unsigned int>(4, "1", 99), { 1, 1, 1, 1 });
  ExpectVectorEqual(GetVectorFromString<unsigned int>(3, "9 8 7 6", 0), { 9, 8, 7 });
  EXPECT_THROW(GetVectorFromString<unsigned int>("-1", 0), itk::ExceptionObject);                  // negative value
  EXPECT_THROW(GetVectorFromString<unsigned int>("9999999999999999999", 0), itk::ExceptionObject); // overflow
  EXPECT_THROW(GetVectorFromString<unsigned int>("1a", 0), itk::ExceptionObject);                  // parse fail

  // ---------- float ----------
  ExpectVectorEqual(GetVectorFromString<float>("1.5 -2.5", 0.0f), { 1.5f, -2.5f });
  ExpectVectorEqual(GetVectorFromString<float>("", 42.0f), { 42.0f });
  ExpectVectorEqual(GetVectorFromString<float>(4, "3.14", 0.0f), { 3.14f, 3.14f, 3.14f, 3.14f });
  EXPECT_THROW(GetVectorFromString<float>("1e1000", 0.0f), itk::ExceptionObject);  // overflow
  EXPECT_THROW(GetVectorFromString<float>("-1e1000", 0.0f), itk::ExceptionObject); // underflow
  EXPECT_THROW(GetVectorFromString<float>("nan abc", 0.0f), itk::ExceptionObject); // parse fail
  EXPECT_THROW(GetVectorFromString<float>("hello", 0.0f), itk::ExceptionObject);   // not numeric

  // ---------- bool ----------
  ExpectVectorEqual(GetVectorFromString<bool>("1 0", false), { true, false });
  ExpectVectorEqual(GetBooleanVectorFromString("10", false), { true, false });
  EXPECT_THROW(GetVectorFromString<bool>("maybe", false), itk::ExceptionObject); // parse fail
  EXPECT_THROW(GetVectorFromString<bool>("10", false), itk::ExceptionObject);    // not space-separated
}

GTEST_TEST(groupStrByDimensions, Basic)
{
  std::vector<std::string> resultVec = groupStrByDimensions("5 5 5 8 6", { 3, 3 });

  EXPECT_EQ(resultVec.size(), 2);
  EXPECT_EQ(resultVec[0], "5 5 5");
  EXPECT_EQ(resultVec[1], "8 6");

  resultVec = groupStrByDimensions("5 5", { 3, 3 });

  EXPECT_EQ(resultVec.size(), 2);
  EXPECT_EQ(resultVec[0], "5 5");
  EXPECT_EQ(resultVec[1], "");

  resultVec = groupStrByDimensions("5 5 6 8 9 10 11 12", { 3, 3 });

  EXPECT_EQ(resultVec.size(), 2);
  EXPECT_EQ(resultVec[0], "5 5 6");
  EXPECT_EQ(resultVec[1], "8 9 10");
}

GTEST_TEST(FormatParameterStringByDimensionAndLevel, Basic)
{
  // Create and initialize the elastix configuration
  auto                                               config = elastix::Configuration::New();
  elastix::Configuration::CommandLineArgumentMapType argMap;
  itk::ParameterFileParser::ParameterMapType         parameterMap;

  // --- First case: PatchSize as unsigned int ---

  // Set PatchSize for 3 resolution levels
  parameterMap["ImpactPatchSize"] = { "5", "5", "5", "11 11 11 29 29", "13", "13" };
  config->Initialize(argMap, parameterMap);

  // Level 0: read 3 individual values → expect "5 5 5"
  std::string resultStr =
    formatParameterStringByDimensionAndLevel<elastix::Configuration>(config.GetPointer(), "Impact", "PatchSize", 0, 3);
  EXPECT_EQ(resultStr, "5 5 5");

  // Level 1: one line with 5 values → "11 11 11 29 29"
  resultStr =
    formatParameterStringByDimensionAndLevel<elastix::Configuration>(config.GetPointer(), "Impact", "PatchSize", 1, 3);
  EXPECT_EQ(resultStr, "11 11 11 29 29");

  // Level 2: only "13" twice → expect fallback to fill 3 values
  resultStr =
    formatParameterStringByDimensionAndLevel<elastix::Configuration>(config.GetPointer(), "Impact", "PatchSize", 2, 3);
  EXPECT_EQ(resultStr, "13 13");

  // --- Second case: PatchSize as float ---

  parameterMap["ImpactPatchSize"] = { "6.0", "6.0", "3.0 3.0 1.5 1.5 1.5", "1.0", "1.0" };
  config->Initialize(argMap, parameterMap);

  // Level 0: two values → repeat to fill 3
  resultStr =
    formatParameterStringByDimensionAndLevel<elastix::Configuration>(config.GetPointer(), "Impact", "PatchSize", 0, 3);
  EXPECT_EQ(resultStr, "6.0 6.0");

  // Level 1: only "1.0 1.0"
  resultStr =
    formatParameterStringByDimensionAndLevel<elastix::Configuration>(config.GetPointer(), "Impact", "PatchSize", 1, 3);
  EXPECT_EQ(resultStr, "1.0 1.0");

  // --- Third case: all entries are "6.0", just checking fallback filling ---

  parameterMap["ImpactPatchSize"] = { "6.0", "6.0", "6.0", "6.0", "3.0 3.0 1.5 1.5 1.5", "1.0", "1.0" };
  config->Initialize(argMap, parameterMap);

  resultStr =
    formatParameterStringByDimensionAndLevel<elastix::Configuration>(config.GetPointer(), "Impact", "PatchSize", 0, 3);
  EXPECT_EQ(resultStr, "6.0 6.0 6.0");

  // Check fallback with single value
  resultStr =
    formatParameterStringByDimensionAndLevel<elastix::Configuration>(config.GetPointer(), "Impact", "PatchSize", 1, 3);
  EXPECT_EQ(resultStr, "6.0");

  resultStr =
    formatParameterStringByDimensionAndLevel<elastix::Configuration>(config.GetPointer(), "Impact", "PatchSize", 2, 3);
  EXPECT_EQ(resultStr, "1.0 1.0");

  // --- Fourth case: test imageDimension auto-detection via ImpactDimension ---

  parameterMap["ImpactDimension"] = { "2", "3", "2 3" };
  parameterMap["ImpactPatchSize"] = { "5", "5", "13", "13", "13", "11 11 11 29 29" };
  config->Initialize(argMap, parameterMap);

  resultStr =
    formatParameterStringByDimensionAndLevel<elastix::Configuration>(config.GetPointer(), "Impact", "PatchSize", 0);
  EXPECT_EQ(resultStr, "5 5");

  resultStr =
    formatParameterStringByDimensionAndLevel<elastix::Configuration>(config.GetPointer(), "Impact", "PatchSize", 1);
  EXPECT_EQ(resultStr, "13 13 13");

  resultStr =
    formatParameterStringByDimensionAndLevel<elastix::Configuration>(config.GetPointer(), "Impact", "PatchSize", 2);
  EXPECT_EQ(resultStr, "11 11 11 29 29");
}

GTEST_TEST(FormatParameterStringByDimensionAndLevel, EarlyStopDueToMissingParam)
{
  auto config = elastix::Configuration::New();

  elastix::Configuration::CommandLineArgumentMapType argMap;
  itk::ParameterFileParser::ParameterMapType         parameterMap;

  // Only 2 values provided instead of 3
  parameterMap["ImpactPatchSize"] = { "5", "5" };
  config->Initialize(argMap, parameterMap);

  std::string resultStr =
    formatParameterStringByDimensionAndLevel<elastix::Configuration>(config.GetPointer(), "Impact", "PatchSize", 0, 3);

  // Should stop early and return "5 5" (not enough to fill 3)
  EXPECT_EQ(resultStr, "5 5");
}

GTEST_TEST(FormatParameterStringByDimensionAndLevel, LevelBeyondAvailable)
{
  auto                                               config = elastix::Configuration::New();
  elastix::Configuration::CommandLineArgumentMapType argMap;
  itk::ParameterFileParser::ParameterMapType         parameterMap;

  parameterMap["ImpactPatchSize"] = { "5", "5" }; // Only 2 values
  config->Initialize(argMap, parameterMap);

  std::string resultStr = formatParameterStringByDimensionAndLevel<elastix::Configuration>(
    config.GetPointer(), "Impact", "PatchSize", 5, 3); // ask level 5
  EXPECT_EQ(resultStr, "");
}
