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

#ifndef elxMovingImagePyramidBase_hxx
#define elxMovingImagePyramidBase_hxx

#include "elxMovingImagePyramidBase.h"
#include <itkDeref.h>

#ifndef ELX_NO_FILESYSTEM_ACCESS
#  include "itkImageFileCastWriter.h"
#endif

namespace elastix
{

/**
 * ******************* BeforeRegistrationBase *******************
 */

template <typename TElastix>
void
MovingImagePyramidBase<TElastix>::BeforeRegistrationBase()
{
  /** Call SetMovingSchedule.*/
  this->SetMovingSchedule();

} // end BeforeRegistrationBase()


/**
 * ******************* BeforeEachResolutionBase *******************
 */

template <typename TElastix>
void
MovingImagePyramidBase<TElastix>::BeforeEachResolutionBase()
{
  /** What is the current resolution level? */
  const unsigned int level = this->m_Registration->GetAsITKBaseType()->GetCurrentLevel();

  const Configuration & configuration = itk::Deref(Superclass::GetConfiguration());

  /** Decide whether or not to write the pyramid images this resolution. */
  bool writePyramidImage = false;
  configuration.ReadParameter(writePyramidImage, "WritePyramidImagesAfterEachResolution", "", level, 0, false);

  /** Get the desired extension / file format. */
  std::string resultImageFormat = "mhd";
  configuration.ReadParameter(resultImageFormat, "ResultImageFormat", 0, false);

  const std::string outputDirectoryPath = configuration.GetCommandLineArgument("-out");

  /** Writing result image. */
  if (writePyramidImage && !outputDirectoryPath.empty())
  {
    /** Create a name for the final result. */
    std::ostringstream makeFileName;
    makeFileName << outputDirectoryPath;
    makeFileName << this->GetComponentLabel() << "." << configuration.GetElastixLevel() << ".R" << level << "."
                 << resultImageFormat;

    /** Save the fixed pyramid image. */
    log::info(std::ostringstream{} << "Writing moving pyramid image " << this->GetComponentLabel()
                                   << " from resolution " << level << "...");
    try
    {
      this->WritePyramidImage(makeFileName.str(), level);
    }
    catch (const itk::ExceptionObject & excp)
    {
      log::error(std::ostringstream{} << "Exception caught: \n" << excp << "Resuming elastix.");
    }
  } // end if

} // end BeforeEachResolutionBase()


/**
 * ********************** SetMovingSchedule **********************
 */

template <typename TElastix>
void
MovingImagePyramidBase<TElastix>::SetMovingSchedule()
{
  /** Get the ImageDimension. */
  const unsigned int ImageDimension = InputImageType::ImageDimension;

  const Configuration & configuration = itk::Deref(Superclass::GetConfiguration());

  /** Read numberOfResolutions. */
  unsigned int numberOfResolutions = 3;
  configuration.ReadParameter(numberOfResolutions, "NumberOfResolutions", 0, true);
  if (numberOfResolutions == 0)
  {
    itkExceptionMacro("The NumberOfResolutions parameter must have a non-zero value!");
  }

  /** Create a default schedule. Set the numberOfLevels first. */
  this->GetAsITKBaseType()->SetNumberOfLevels(numberOfResolutions);
  ScheduleType schedule = this->GetAsITKBaseType()->GetSchedule();

  /** Set the movingPyramidSchedule to the MovingImagePyramidSchedule given
   * in the parameter-file. The following parameter file fields can be used:
   * ImagePyramidSchedule
   * MovingImagePyramidSchedule
   * MovingImagePyramid<i>Schedule, for the i-th moving image pyramid used.
   */
  bool found = true;
  for (unsigned int i = 0; i < numberOfResolutions; ++i)
  {
    for (unsigned int j = 0; j < ImageDimension; ++j)
    {
      bool               ijfound = false;
      const unsigned int entrynr = i * ImageDimension + j;
      ijfound |= configuration.ReadParameter(schedule[i][j], "ImagePyramidSchedule", entrynr, false);
      ijfound |= configuration.ReadParameter(schedule[i][j], "MovingImagePyramidSchedule", entrynr, false);
      ijfound |= configuration.ReadParameter(schedule[i][j], "Schedule", this->GetComponentLabel(), entrynr, -1, false);

      /** Remember if for at least one schedule element no value could be found. */
      found &= ijfound;

    } // end for ImageDimension
  } // end for numberOfResolutions

  if (!found && configuration.GetPrintErrorMessages())
  {
    log::warn(std::ostringstream{} << "WARNING: the moving pyramid schedule is not fully specified!\n"
                                   << "  A default pyramid schedule is used.");
  }
  else
  {
    /** Set the schedule into this class. */
    this->GetAsITKBaseType()->SetSchedule(schedule);
  }

} // end SetMovingSchedule()


/*
 * ******************* WritePyramidImage ********************
 */

template <typename TElastix>
void
MovingImagePyramidBase<TElastix>::WritePyramidImage(const std::string & filename,
                                                    const unsigned int  level) // const
{
  const Configuration & configuration = itk::Deref(Superclass::GetConfiguration());

  /** Read output pixeltype from parameter the file. Replace possible " " with "_". */
  std::string resultImagePixelType = "short";
  configuration.ReadParameter(resultImagePixelType, "ResultImagePixelType", 0, false);
  const std::string::size_type pos = resultImagePixelType.find(" ");
  if (pos != std::string::npos)
  {
    resultImagePixelType.replace(pos, 1, "_");
  }

  /** Read from the parameter file if compression is desired. */
  bool doCompression = false;
  configuration.ReadParameter(doCompression, "CompressResultImage", 0, false);

  /** Do the writing. */
  log::to_stdout("  Writing moving pyramid image ...");
#ifndef ELX_NO_FILESYSTEM_ACCESS
  try
  {
    itk::WriteCastedImage(*(this->GetAsITKBaseType()->GetOutput(level)), filename, resultImagePixelType, doCompression);
  }
  catch (itk::ExceptionObject & excp)
  {
#else
  // Always throw -- do not include support code or access filesystem with wasm
  itk::ExceptionObject excp;
#endif
    /** Add information to the exception. */
    excp.SetLocation("MovingImagePyramidBase - BeforeEachResolutionBase()");
    std::string err_str = excp.GetDescription();
    err_str += "\nError occurred while writing pyramid image.\n";
    excp.SetDescription(err_str);

    /** Pass the exception to an higher level. */
#ifndef ELX_NO_FILESYSTEM_ACCESS
    throw;
  }
#else
  throw excp;
#endif

} // end WritePyramidImage()


} // end namespace elastix

#endif // end #ifndef elxMovingImagePyramidBase_hxx
