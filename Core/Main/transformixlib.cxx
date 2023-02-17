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

#include "transformixlib.h"
#include "elastix.h" // For ConvertSecondsToDHMS and GetCurrentDateAndTime.

#ifdef _ELASTIX_USE_MEVISDICOMTIFF
#  include "itkUseMevisDicomTiff.h"
#endif

#include "elxTransformixMain.h"
#include <iostream>
#include <string>
#include <vector>
#include <queue>
#include <ctime>

#include "itkObject.h"
#include "itkDataObject.h"
#include <itksys/SystemTools.hxx>
#include <itksys/SystemInformation.hxx>

#include "itkTimeProbe.h"

namespace transformix
{

/**
 * ******************* Constructor ***********************
 */

TRANSFORMIX::TRANSFORMIX()
  : m_ResultImage(nullptr)
{} // end Constructor


/**
 * ******************* Destructor ***********************
 */

TRANSFORMIX::~TRANSFORMIX()
{
  this->m_ResultImage = nullptr;
} // end Destructor


/**
 * ******************* GetResultImage ***********************
 */

TRANSFORMIX::ConstImagePointer
TRANSFORMIX::GetResultImage() const
{
  return this->m_ResultImage;
} // end GetResultImage()


TRANSFORMIX::ImagePointer
TRANSFORMIX::GetResultImage()
{
  return this->m_ResultImage;
} // end GetResultImage()


/**
 * ******************* TransformImage ***********************
 */

int
TRANSFORMIX::TransformImage(ImagePointer                    inputImage,
                            std::vector<ParameterMapType> & parameterMaps,
                            const std::string &             outputPath,
                            bool                            performLogging,
                            bool                            performCout)
{
  /** Some typedef's.*/
  using TransformixMainType = elx::TransformixMain;
  using TransformixMainPointer = TransformixMainType::Pointer;
  using ArgumentMapType = TransformixMainType::ArgumentMapType;
  using ArgumentMapEntryType = ArgumentMapType::value_type;
  using DataObjectContainerType = elx::MainBase::DataObjectContainerType;
  using DataObjectContainerPointer = elx::MainBase::DataObjectContainerPointer;

  /** Declare an instance of the Transformix class. */
  TransformixMainPointer transformix;

  DataObjectContainerPointer movingImageContainer = nullptr;
  ;
  DataObjectContainerPointer resultImageContainer = nullptr;

  /** Initialize. */
  int             returndummy = 0;
  ArgumentMapType argMap;
  bool            outFolderPresent = false;
  std::string     outFolder = "";
  std::string     logFileName = "";

  std::string key;
  std::string value;

  if (!outputPath.empty())
  {
    key = "-out";
    value = outputPath;

    /** Make sure that last character of the output folder equals a '/'. */
    if (outputPath.back() != '/')
    {
      value.append("/");
    }

    outFolderPresent = true;
  }
  else
  {
    /** Put command line parameters into parameterFileList. */
    // there must be an "-out", this is checked later in code!!
    key = "-out";
    value = "output_path_not_set";
  }

  /** Save this information. */
  outFolder = value;

  /** Attempt to save the arguments in the ArgumentMap. */
  if (argMap.count(key) == 0)
  {
    argMap.insert(ArgumentMapEntryType(key.c_str(), value.c_str()));
  }
  else if (performCout)
  {
    /** Duplicate arguments. */
    std::cerr << "WARNING!\n"
              << "Argument " << key.c_str() << "is only required once.\n"
              << "Arguments " << key.c_str() << " " << value.c_str() << "are ignored" << std::endl;
  }

  if (performLogging)
  {
    /** Check if the output directory exists. */
    bool outFolderExists = itksys::SystemTools::FileIsDirectory(outFolder);
    if (!outFolderExists)
    {
      if (performCout)
      {
        std::cerr << "ERROR: the output directory does not exist.\n"
                  << "You are responsible for creating it." << std::endl;
      }
      return (-2);
    }
    else
    {
      if (performLogging)
      {
        logFileName = outFolder + "transformix.log";
      }
    }
  }

  /** The argv0 argument, required for finding the component.dll/so's. */
  argMap.insert(ArgumentMapEntryType("-argv0", "transformix"));

  /** Setup the log system. */
  const elx::log::guard logGuard{};
  int                   returndummy2 = elx::log::setup(logFileName, performLogging, performCout) ? 0 : 1;
  if (returndummy2 && performCout)
  {
    if (performCout)
    {
      std::cerr << "ERROR while setting up the log system." << std::endl;
    }
    return (returndummy2);
  }
  elx::log::info("");

  /** Declare a timer, start it and print the start time. */
  itk::TimeProbe totaltimer;
  totaltimer.Start();
  elx::log::info(std::ostringstream{} << "transformix is started at " << GetCurrentDateAndTime() << ".\n");

  /**
   * ********************* START TRANSFORMATION *******************
   */

  /** Set transformix. */
  transformix = TransformixMainType::New();

  /** Set stuff from input or needed for output */
  movingImageContainer = DataObjectContainerType::New();
  movingImageContainer->CreateElementAt(0) = inputImage;
  transformix->SetMovingImageContainer(movingImageContainer);
  transformix->SetResultImageContainer(resultImageContainer);

  /** Run transformix. */
  returndummy = transformix->Run(argMap, parameterMaps);

  /** Check if transformix run without errors. */
  if (returndummy != 0)
  {
    elx::log::error("Errors occurred");
    return returndummy;
  }

  /** Get the result image */
  resultImageContainer = transformix->GetModifiableResultImageContainer();

  /** Stop timer and print it. */
  totaltimer.Stop();
  elx::log::info(std::ostringstream{} << "\nTransformix has finished at " << GetCurrentDateAndTime() << ".\n"
                                      << "Elapsed time: " << ConvertSecondsToDHMS(totaltimer.GetMean(), 1) << ".\n");

  this->m_ResultImage = resultImageContainer->ElementAt(0);

  /** Clean up. */
  transformix = nullptr;

  /** Exit and return the error code. */
  return returndummy;

} // end TransformImage()


/**
 * ******************* TransformImage ***********************
 */

int
TRANSFORMIX::TransformImage(ImagePointer       inputImage,
                            ParameterMapType & parameterMap,
                            std::string        outputPath,
                            bool               performLogging,
                            bool               performCout)
{
  // Transform single parameter map to a one-sized vector of parameter maps and call other
  // transform method.
  std::vector<ParameterMapType> parameterMaps;
  parameterMaps.push_back(parameterMap);
  return TransformImage(inputImage, parameterMaps, outputPath, performLogging, performCout);
} // end TransformImage()

} // namespace transformix
