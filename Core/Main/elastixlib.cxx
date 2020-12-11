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

// Elastix header files:
#include "elastixlib.h"
#include "elxElastixMain.h"
#include "elastix.h" // For ConvertSecondsToDHMS and GetCurrentDateAndTime.

#ifdef _ELASTIX_USE_MEVISDICOMTIFF
#  include "itkUseMevisDicomTiff.h"
#endif

// ITK header files:
#include <itkDataObject.h>
#include <itkObject.h>
#include <itkTimeProbe.h>
#include <itksys/SystemInformation.hxx>
#include <itksys/SystemTools.hxx>

// Standard C++ header files:
#include <cassert>
#include <climits> // For UINT_MAX.
#include <iostream>
#include <string>
#include <vector>

namespace elastix
{

/**
 * ******************* Constructor ***********************
 */

ELASTIX::ELASTIX()
{
  assert(BaseComponent::IsElastixLibrary());
}

/**
 * ******************* Destructor ***********************
 */

ELASTIX::~ELASTIX()
{
  assert(BaseComponent::IsElastixLibrary());
}

/**
 * ******************* GetResultImage ***********************
 */

ELASTIX::ConstImagePointer
ELASTIX::GetResultImage(void) const
{
  return this->m_ResultImage;
} // end GetResultImage()


ELASTIX::ImagePointer
ELASTIX::GetResultImage(void)
{
  return this->m_ResultImage;
} // end GetResultImage()


/**
 * ******************* GetTransformParameterMap ***********************
 */

ELASTIX::ParameterMapType
ELASTIX::GetTransformParameterMap(void) const
{
  return this->m_TransformParametersList.back();
} // end GetTransformParameterMap()


/**
 * ******************* GetTransformParameterMapList ***********************
 */

ELASTIX::ParameterMapListType
ELASTIX::GetTransformParameterMapList(void) const
{
  return this->m_TransformParametersList;
} // end GetTransformParameterMapList()


/**
 * ******************* RegisterImages ***********************
 */

int
ELASTIX::RegisterImages(ImagePointer             fixedImage,
                        ImagePointer             movingImage,
                        const ParameterMapType & parameterMap,
                        const std::string &      outputPath,
                        bool                     performLogging,
                        bool                     performCout,
                        ImagePointer             fixedMask,
                        ImagePointer             movingMask)
{
  std::vector<ParameterMapType> parameterMaps(1);
  parameterMaps[0] = parameterMap;
  return this->RegisterImages(
    fixedImage, movingImage, parameterMaps, outputPath, performLogging, performCout, fixedMask, movingMask);

} // end RegisterImages()


/**
 * ******************* RegisterImages ***********************
 */

int
ELASTIX::RegisterImages(ImagePointer                          fixedImage,
                        ImagePointer                          movingImage,
                        const std::vector<ParameterMapType> & parameterMaps,
                        const std::string &                   outputPath,
                        bool                                  performLogging,
                        bool                                  performCout,
                        ImagePointer                          fixedMask,
                        ImagePointer                          movingMask,
                        ObjectPointer                         transform)
{
  /** Some typedef's. */
  typedef elx::ElastixMain                            ElastixMainType;
  typedef ElastixMainType::DataObjectContainerType    DataObjectContainerType;
  typedef ElastixMainType::DataObjectContainerPointer DataObjectContainerPointer;
  typedef ElastixMainType::FlatDirectionCosinesType   FlatDirectionCosinesType;

  typedef ElastixMainType::ArgumentMapType ArgumentMapType;
  typedef ArgumentMapType::value_type      ArgumentMapEntryType;

  // Clear output transform parameters
  this->m_TransformParametersList.clear();

  /** Some declarations and initialisations. */

  std::string value;

  /** Setup the argumentMap for output path. */
  if (!outputPath.empty())
  {
    /** Put command line parameters into parameterFileList. */
    value = outputPath;

    /** Make sure that last character of the output folder equals a '/'. */
    if (value.find_last_of("/") != value.size() - 1)
    {
      value.append("/");
    }
  }
  else
  {
    /** Put command line parameters into parameterFileList. */
    // there must be an "-out", this is checked later in code!!
    value = "output_path_not_set";
  }

  /** Save this information. */
  const auto outFolder = value;

  const ArgumentMapType argMap{ /** The argv0 argument, required for finding the component.dll/so's. */
                                ArgumentMapEntryType("-argv0", "elastix"),
                                ArgumentMapEntryType("-out", outFolder)
  };

  /** Check if the output directory exists. */
  if (performLogging && !itksys::SystemTools::FileIsDirectory(outFolder))
  {
    if (performCout)
    {
      std::cerr << "ERROR: the output directory does not exist." << std::endl;
      std::cerr << "You are responsible for creating it." << std::endl;
    }
    return -2;
  }

  /** Setup xout. */
  const std::string      logFileName = performLogging ? (outFolder + "elastix.log") : "";
  const elx::xoutManager manager{};
  int                    returndummy = elx::xoutSetup(logFileName.c_str(), performLogging, performCout);
  if ((returndummy != 0) && performCout)
  {
    if (performCout)
    {
      std::cerr << "ERROR while setting up xout." << std::endl;
    }
    return returndummy;
  }
  elxout << std::endl;

  /** Declare a timer, start it and print the start time. */
  itk::TimeProbe totaltimer;
  totaltimer.Start();
  elxout << "elastix is started at " << GetCurrentDateAndTime() << ".\n" << std::endl;

  /************************************************************************
   *                                              *
   *  Generate containers with input images       *
   *                                              *
   ************************************************************************/

  /* Allocate and store images in containers */
  auto fixedImageContainer = DataObjectContainerType::New();
  auto movingImageContainer = DataObjectContainerType::New();
  fixedImageContainer->CreateElementAt(0) = fixedImage;
  movingImageContainer->CreateElementAt(0) = movingImage;

  DataObjectContainerPointer fixedMaskContainer = nullptr;
  DataObjectContainerPointer movingMaskContainer = nullptr;
  DataObjectContainerPointer resultImageContainer = nullptr;
  FlatDirectionCosinesType   fixedImageOriginalDirection;

  /* Allocate and store masks in containers if available*/
  if (fixedMask)
  {
    fixedMaskContainer = DataObjectContainerType::New();
    fixedMaskContainer->CreateElementAt(0) = fixedMask;
  }
  if (movingMask)
  {
    movingMaskContainer = DataObjectContainerType::New();
    movingMaskContainer->CreateElementAt(0) = movingMask;
  }

  // todo original direction cosin, problem is that Image type is unknown at this in elastixlib.cxx
  // for now in elaxElastixTemplate (Run()) direction cosines are taken from fixed image

  /**
   * ********************* START REGISTRATION *********************
   *
   * Do the (possibly multiple) registration(s).
   */

  const auto nrOfParameterFiles = parameterMaps.size();
  assert(nrOfParameterFiles <= UINT_MAX);

  for (unsigned i{}; i < static_cast<unsigned>(nrOfParameterFiles); ++i)
  {
    /** Create another instance of ElastixMain. */
    const auto elastixMain = ElastixMainType::New();

    /** Set stuff we get from a former registration. */
    elastixMain->SetInitialTransform(transform);
    elastixMain->SetFixedImageContainer(fixedImageContainer);
    elastixMain->SetMovingImageContainer(movingImageContainer);
    elastixMain->SetFixedMaskContainer(fixedMaskContainer);
    elastixMain->SetMovingMaskContainer(movingMaskContainer);
    elastixMain->SetResultImageContainer(resultImageContainer);
    elastixMain->SetOriginalFixedImageDirectionFlat(fixedImageOriginalDirection);

    /** Set the current elastix-level. */
    elastixMain->SetElastixLevel(i);
    elastixMain->SetTotalNumberOfElastixLevels(nrOfParameterFiles);

    /** Print a start message. */
    elxout << "-------------------------------------------------------------------------"
           << "\n"
           << std::endl;
    elxout << "Running elastix with parameter map " << i << std::endl;

    /** Declare a timer, start it and print the start time. */
    itk::TimeProbe timer;
    timer.Start();
    elxout << "Current time: " << GetCurrentDateAndTime() << "." << std::endl;

    /** Start registration. */
    returndummy = elastixMain->Run(argMap, parameterMaps[i]);

    /** Check for errors. */
    if (returndummy != 0)
    {
      xl::xout["error"] << "Errors occurred!" << std::endl;
      return returndummy;
    }

    /** Get the transform, the fixedImage and the movingImage
     * in order to put it in the (possibly) next registration.
     */
    transform = elastixMain->GetModifiableFinalTransform();
    fixedImageContainer = elastixMain->GetModifiableFixedImageContainer();
    movingImageContainer = elastixMain->GetModifiableMovingImageContainer();
    fixedMaskContainer = elastixMain->GetModifiableFixedMaskContainer();
    movingMaskContainer = elastixMain->GetModifiableMovingMaskContainer();
    resultImageContainer = elastixMain->GetModifiableResultImageContainer();
    fixedImageOriginalDirection = elastixMain->GetOriginalFixedImageDirectionFlat();

    /** Stop timer and print it. */
    timer.Stop();
    elxout << "\nCurrent time: " << GetCurrentDateAndTime() << "." << std::endl;
    elxout << "Time used for running elastix with this parameter file: " << ConvertSecondsToDHMS(timer.GetMean(), 1)
           << ".\n"
           << std::endl;

    /** Get the transformation parameter map. */
    this->m_TransformParametersList.push_back(elastixMain->GetTransformParametersMap());

    /** Set initial transform to an index number instead of a parameter filename. */
    if (i > 0)
    {
      std::stringstream toString;
      toString << (i - 1);
      this->m_TransformParametersList[i]["InitialTransformParametersFileName"][0] = toString.str();
    }
  } // end loop over registrations

  elxout << "-------------------------------------------------------------------------"
         << "\n"
         << std::endl;

  /** Stop totaltimer and print it. */
  totaltimer.Stop();
  elxout << "Total time elapsed: " << ConvertSecondsToDHMS(totaltimer.GetMean(), 1) << ".\n" << std::endl;

  /************************************************************************
   *                                *
   *  Cleanup everything            *
   *                                *
   ************************************************************************/

  /*
   *  Make sure all the components that are defined in a Module (.DLL/.so)
   *  are deleted before the modules are closed.
   */

  /* Set result image for output */
  if (resultImageContainer.IsNotNull() && resultImageContainer->Size() > 0 &&
      resultImageContainer->ElementAt(0).IsNotNull())
  {
    this->m_ResultImage = resultImageContainer->ElementAt(0);
  }

  transform = nullptr;
  fixedImageContainer = nullptr;
  movingImageContainer = nullptr;
  fixedMaskContainer = nullptr;
  movingMaskContainer = nullptr;
  resultImageContainer = nullptr;

  /** Close the modules. */
  ElastixMainType::UnloadComponents();

  /** Exit and return the error code. */
  return 0;

} // end RegisterImages()


} // end namespace elastix
