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
#include "elxElastixBase.h"
#include <Core/elxVersionMacros.h>
#include "elxConversion.h"
#include <sstream>
#include "itkMersenneTwisterRandomVariateGenerator.h"

namespace elastix
{

namespace
{

/**
 * ********************* GenerateFileNameContainer ******************
 */

/** Read a series of command line options that satisfy the following syntax:
 * {-f,-f0} \<filename0\> [-f1 \<filename1\> [ -f2 \<filename2\> ... ] ]
 *
 * This function is used by BeforeAllBase, and is not meant be used
 * at other locations. The errorcode remains the input value if no errors
 * occur. It is set to errorcode | 1 if the option was not given.
 */
ElastixBase::FileNameContainerPointer
GenerateFileNameContainer(const Configuration & configuration,
                          const std::string &   optionkey,
                          int &                 errorcode,
                          bool                  printerrors,
                          bool                  printinfo)
{
  const auto fileNameContainer = ElastixBase::FileNameContainerType::New();

  /** Try optionkey0. */
  std::ostringstream argusedss;
  argusedss << optionkey << 0;
  std::string argused = argusedss.str();
  std::string check = configuration.GetCommandLineArgument(argused);
  if (check.empty())
  {
    /** Try optionkey. */
    std::ostringstream argusedss2;
    argusedss2 << optionkey;
    argused = argusedss2.str();
    check = configuration.GetCommandLineArgument(argused);
    if (check.empty())
    {
      /** Both failed; return an error message, if desired. */
      if (printerrors)
      {
        log::error(std::ostringstream{} << "ERROR: No CommandLine option \"" << optionkey << "\" or \"" << optionkey
                                        << 0 << "\" given!");
      }
      errorcode |= 1;

      return fileNameContainer;
    }
  }

  /** Optionkey or optionkey0 is found. */
  if (!check.empty())
  {
    /** Print info, if desired. */
    if (printinfo)
    {
      /** Print the option, with some spaces, followed by the value. */
      int          nrSpaces0 = 10 - argused.length();
      unsigned int nrSpaces = nrSpaces0 > 1 ? nrSpaces0 : 1;
      std::string  spaces = "";
      spaces.resize(nrSpaces, ' ');
      log::info(std::ostringstream{} << argused << spaces << check);
    }
    fileNameContainer->CreateElementAt(0) = check;

    /** Loop over all optionkey<i> options given with i > 0. */
    unsigned int i = 1;
    bool         readsuccess = true;
    while (readsuccess)
    {
      std::ostringstream argusedss2;
      argusedss2 << optionkey << i;
      argused = argusedss2.str();
      check = configuration.GetCommandLineArgument(argused);
      if (check.empty())
      {
        readsuccess = false;
      }
      else
      {
        if (printinfo)
        {
          /** Print the option, with some spaces, followed by the value. */
          int          nrSpaces0 = 10 - argused.length();
          unsigned int nrSpaces = nrSpaces0 > 1 ? nrSpaces0 : 1;
          std::string  spaces = "";
          spaces.resize(nrSpaces, ' ');
          log::info(std::ostringstream{} << argused << spaces << check);
        }
        fileNameContainer->CreateElementAt(i) = check;
        ++i;
      }
    } // end while
  }   // end if

  return fileNameContainer;

} // end GenerateFileNameContainer()

} // end unnamed namespace


/**
 * ********************* Constructor ****************************
 */

ElastixBase::ElastixBase() = default;

/**
 * ********************* GenerateDataObjectContainer ***********************
 */

ElastixBase::DataObjectContainerPointer
ElastixBase::GenerateDataObjectContainer(DataObjectPointer dataObject)
{
  /** Allocate container pointer. */
  const auto container = DataObjectContainerType::New();

  /** Store object in container. */
  container->push_back(dataObject);

  /** Return the pointer to the new container. */
  return container;
}


/**
 * ********************* SetDBIndex ***********************
 */

void
ElastixBase::SetDBIndex(DBIndexType _arg)
{
  /** If m_DBIndex is not set, set it. */
  if (m_DBIndex != _arg)
  {
    m_DBIndex = _arg;
    this->itk::Object::Modified();
  }

} // end SetDBIndex()


/**
 * ************************ BeforeAllBase ***************************
 */

int
ElastixBase::BeforeAllBase()
{
  /** Declare the return value and initialize it. */
  int returndummy = 0;

  /** Check Command line options and print them to the logfile. */
  log::info("ELASTIX version: " ELASTIX_VERSION_STRING "\nCommand line options from ElastixBase:");

  if (!BaseComponent::IsElastixLibrary())
  {
    // Note: The filenames of the input images and the masks are only relevant to the elastix executable, not the
    // library.

    /** Read the fixed and moving image filenames. These are obliged options,
     * so print an error if they are not present.
     * Print also some info (second boolean = true).
     */
    m_FixedImageFileNameContainer = GenerateFileNameContainer(*m_Configuration, "-f", returndummy, true, true);
    m_MovingImageFileNameContainer = GenerateFileNameContainer(*m_Configuration, "-m", returndummy, true, true);

    /** Read the fixed and moving mask filenames. These are not obliged options,
     * so do not print any errors if they are not present.
     * Do print some info (second boolean = true).
     */
    int maskreturndummy = 0;
    m_FixedMaskFileNameContainer = GenerateFileNameContainer(*m_Configuration, "-fMask", maskreturndummy, false, true);
    if (maskreturndummy != 0)
    {
      log::info("-fMask    unspecified, so no fixed mask used");
    }
    maskreturndummy = 0;
    m_MovingMaskFileNameContainer = GenerateFileNameContainer(*m_Configuration, "-mMask", maskreturndummy, false, true);
    if (maskreturndummy != 0)
    {
      log::info("-mMask    unspecified, so no moving mask used");
    }
  }

  /** Check for appearance of "-out". */
  if (const std::string check = m_Configuration->GetCommandLineArgument("-out"); !check.empty())
  {
    /** Make sure that last character of the output folder equals a '/' or '\'. */
    std::string folder(check);
    const char  last = folder.back();
    if (last != '/' && last != '\\')
    {
      folder.append("/");
      folder = Conversion::ToNativePathNameSeparators(folder);

      m_Configuration->SetCommandLineArgument("-out", folder);
    }
    log::info(std::ostringstream{} << "-out      " << check);
  }

  /** Print all "-p". */
  unsigned int i = 1;
  bool         loop = true;
  while (loop)
  {
    std::ostringstream tempPname;
    tempPname << "-p(" << i << ")";

    if (const std::string check = m_Configuration->GetCommandLineArgument(tempPname.str()); check.empty())
    {
      loop = false;
    }
    else
    {
      log::info(std::ostringstream{} << "-p        " << check);
    }
    ++i;
  }

  /** Check for appearance of "-priority", if this is a Windows station. */
#ifdef _WIN32
  if (const std::string check = m_Configuration->GetCommandLineArgument("-priority"); check.empty())
  {
    log::info("-priority unspecified, so NORMAL process priority");
  }
  else
  {
    log::info(std::ostringstream{} << "-priority " << check);
  }
#endif

  /** Check for appearance of -threads, which specifies the maximum number of threads. */
  if (const std::string check = m_Configuration->GetCommandLineArgument("-threads"); check.empty())
  {
    log::info("-threads  unspecified, so all available threads are used");
  }
  else
  {
    log::info(std::ostringstream{} << "-threads  " << check);
  }

  /** Check the very important UseDirectionCosines parameter. */
  bool retudc = m_Configuration->ReadParameter(m_UseDirectionCosines, "UseDirectionCosines", 0);
  if (!retudc)
  {
    log::warn(
      std::ostringstream{} << "\nWARNING: The option \"UseDirectionCosines\" was not found in your parameter file.\n"
                           << "  From elastix 4.8 it defaults to true!\n"
                           << "This may change the behavior of your registrations considerably.\n");
  }

  /** Set the random seed. Use 121212 as a default, which is the same as
   * the default in the MersenneTwister code.
   * Use silent parameter file readout, to avoid annoying warning when
   * starting elastix */
  using RandomGeneratorType = itk::Statistics::MersenneTwisterRandomVariateGenerator;
  using SeedType = RandomGeneratorType::IntegerType;
  unsigned int randomSeed = 121212;
  m_Configuration->ReadParameter(randomSeed, "RandomSeed", 0, false);
  RandomGeneratorType::Pointer randomGenerator = RandomGeneratorType::GetInstance();
  randomGenerator->SetSeed(static_cast<SeedType>(randomSeed));

  /** Return a value. */
  return returndummy;

} // end BeforeAllBase()


/**
 * ************************ BeforeAllTransformixBase ***************************
 */

int
ElastixBase::BeforeAllTransformixBase()
{
  /** Declare the return value and initialize it. */
  int returndummy = 0;

  /** Print to log file. */
  log::info("ELASTIX version: " ELASTIX_VERSION_STRING);

  /** Check Command line options and print them to the logfile. */
  log::info("Command line options from ElastixBase:");
  if (!BaseComponent::IsElastixLibrary())
  {
    /** Read the input image filenames. These are not obliged options,
     * so do not print an error if they are not present.
     * Print also some info (second boolean = true)
     * Save the result in the moving image file name container.
     */
    int inreturndummy = 0;
    m_MovingImageFileNameContainer = GenerateFileNameContainer(*m_Configuration, "-in", inreturndummy, false, true);
    if (inreturndummy != 0)
    {
      log::info("-in       unspecified, so no input image specified");
    }
  }
  /** Check for appearance of "-out". */
  if (const std::string check = m_Configuration->GetCommandLineArgument("-out"); !check.empty())
  {
    /** Make sure that last character of -out equals a '/'. */
    if (check.back() != '/')
    {
      m_Configuration->SetCommandLineArgument("-out", check + '/');
    }
    log::info(std::ostringstream{} << "-out      " << check);
  }

  /** Check for appearance of -threads, which specifies the maximum number of threads. */
  if (const std::string check = m_Configuration->GetCommandLineArgument("-threads"); check.empty())
  {
    log::info("-threads  unspecified, so all available threads are used");
  }
  else
  {
    log::info(std::ostringstream{} << "-threads  " << check);
  }
  if (!BaseComponent::IsElastixLibrary())
  {
    /** Print "-tp". */
    const std::string check = m_Configuration->GetCommandLineArgument("-tp");
    log::info(std::ostringstream{} << "-tp       " << check);
  }
  /** Retrieve the very important UseDirectionCosines parameter. */
  m_Configuration->ReadParameter(m_UseDirectionCosines, "UseDirectionCosines", 0);

  return returndummy;

} // end BeforeAllTransformixBase()


/**
 * ********************** GetResultImage *************************
 */

itk::DataObject *
ElastixBase::GetResultImage(const unsigned int idx) const
{
  if (idx < this->GetNumberOfResultImages())
  {
    return m_ResultImageContainer->ElementAt(idx).GetPointer();
  }

  return nullptr;

} // end GetResultImage()


/**
 * ********************** SetResultImage *************************
 */

void
ElastixBase::SetResultImage(DataObjectPointer result_image)
{
  this->SetResultImageContainer(GenerateDataObjectContainer(result_image));
} // end SetResultImage()


/**
 * ********************** GetResultDeformationField *************************
 */

itk::DataObject *
ElastixBase::GetResultDeformationField(unsigned int idx) const
{
  if (idx < this->GetNumberOfResultDeformationFields())
  {
    return m_ResultDeformationFieldContainer->ElementAt(idx).GetPointer();
  }

  return nullptr;

} // end GetResultDeformationField()

/**
 * ********************** SetResultDeformationField *************************
 */

void
ElastixBase::SetResultDeformationField(DataObjectPointer result_deformationfield)
{
  this->SetResultDeformationFieldContainer(ElastixBase::GenerateDataObjectContainer(result_deformationfield));
} // end SetResultDeformationField()

/**
 * ******************** GetUseDirectionCosines ********************
 */

bool
ElastixBase::GetUseDirectionCosines() const
{
  return m_UseDirectionCosines;
}


/**
 * ******************** SetOriginalFixedImageDirectionFlat ********************
 */

void
ElastixBase::SetOriginalFixedImageDirectionFlat(const FlatDirectionCosinesType & arg)
{
  m_OriginalFixedImageDirectionFlat = arg;
}


/**
 * ******************** GetOriginalFixedImageDirectionFlat ********************
 */

const ElastixBase::FlatDirectionCosinesType &
ElastixBase::GetOriginalFixedImageDirectionFlat() const
{
  return m_OriginalFixedImageDirectionFlat;
}


/**
 * ************** GetTransformParameterMap *****************
 */

itk::ParameterMapInterface::ParameterMapType
ElastixBase::GetTransformParameterMap() const
{
  return m_TransformParameterMap;
} // end GetTransformParameterMap()


/**
 * ************** SetTransformConfigurations *********************
 */

void
ElastixBase::SetTransformConfigurations(const std::vector<Configuration::ConstPointer> & configurations)
{
  m_TransformConfigurations = configurations;
}


/**
 * ************** GetConfiguration *********************
 */

Configuration::ConstPointer
ElastixBase::GetTransformConfiguration(const size_t index) const
{
  return m_TransformConfigurations[index];
}


/**
 * ************** GetPreviousTransformConfiguration *********************
 */

Configuration::ConstPointer
ElastixBase::GetPreviousTransformConfiguration(const Configuration & configuration) const
{
  const auto begin = m_TransformConfigurations.cbegin();
  const auto end = m_TransformConfigurations.cend();
  const auto found = std::find(begin, end, &configuration);

  return (found == begin || found == end) ? nullptr : *(found - 1);
}


/**
 * ************** GetNumberOfTransformConfigurations *********************
 */

size_t
ElastixBase::GetNumberOfTransformConfigurations() const
{
  return m_TransformConfigurations.size();
}


} // end namespace elastix
