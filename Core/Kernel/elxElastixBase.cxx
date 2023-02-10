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
        log::error(log::get_ostringstream()
                   << "ERROR: No CommandLine option \"" << optionkey << "\" or \"" << optionkey << 0 << "\" given!");
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
      log::info(log::get_ostringstream() << argused << spaces << check);
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
          log::info(log::get_ostringstream() << argused << spaces << check);
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
  if (this->m_DBIndex != _arg)
  {
    this->m_DBIndex = _arg;
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

  /** Set the default precision of floating values in the output. */
  this->m_Configuration->ReadParameter(this->m_DefaultOutputPrecision, "DefaultOutputPrecision", 0, false);
  log::set_precision(this->m_DefaultOutputPrecision);

  /** Check Command line options and print them to the logfile. */
  log::info("ELASTIX version: " ELASTIX_VERSION_STRING "\nCommand line options from ElastixBase:");
  std::string check = "";

  /** Read the fixed and moving image filenames. These are obliged options,
   * so print an error if they are not present.
   * Print also some info (second boolean = true).
   */
  if (!BaseComponent::IsElastixLibrary())
  {
    this->m_FixedImageFileNameContainer =
      GenerateFileNameContainer(*(this->m_Configuration), "-f", returndummy, true, true);
    this->m_MovingImageFileNameContainer =
      GenerateFileNameContainer(*(this->m_Configuration), "-m", returndummy, true, true);
  }
  /** Read the fixed and moving mask filenames. These are not obliged options,
   * so do not print any errors if they are not present.
   * Do print some info (second boolean = true).
   */
  int maskreturndummy = 0;
  this->m_FixedMaskFileNameContainer =
    GenerateFileNameContainer(*(this->m_Configuration), "-fMask", maskreturndummy, false, true);
  if (maskreturndummy != 0)
  {
    log::info("-fMask    unspecified, so no fixed mask used");
  }
  maskreturndummy = 0;
  this->m_MovingMaskFileNameContainer =
    GenerateFileNameContainer(*(this->m_Configuration), "-mMask", maskreturndummy, false, true);
  if (maskreturndummy != 0)
  {
    log::info("-mMask    unspecified, so no moving mask used");
  }

  /** Check for appearance of "-out".
   * This check has already been performed in elastix.cxx,
   * Here we do it again. MS: WHY?
   */
  check = this->GetConfiguration()->GetCommandLineArgument("-out");
  if (check.empty())
  {
    log::error(log::get_ostringstream() << "ERROR: No CommandLine option \"-out\" given!");
    returndummy |= 1;
  }
  else
  {
    /** Make sure that last character of the output folder equals a '/' or '\'. */
    std::string folder(check);
    const char  last = folder.back();
    if (last != '/' && last != '\\')
    {
      folder.append("/");
      folder = Conversion::ToNativePathNameSeparators(folder);

      this->GetConfiguration()->SetCommandLineArgument("-out", folder);
    }
    log::info(log::get_ostringstream() << "-out      " << check);
  }

  /** Print all "-p". */
  unsigned int i = 1;
  bool         loop = true;
  while (loop)
  {
    std::ostringstream tempPname;
    tempPname << "-p(" << i << ")";
    check = this->GetConfiguration()->GetCommandLineArgument(tempPname.str());
    if (check.empty())
    {
      loop = false;
    }
    else
    {
      log::info(log::get_ostringstream() << "-p        " << check);
    }
    ++i;
  }

  /** Check for appearance of "-priority", if this is a Windows station. */
#ifdef _WIN32
  check = this->GetConfiguration()->GetCommandLineArgument("-priority");
  if (check.empty())
  {
    log::info("-priority unspecified, so NORMAL process priority");
  }
  else
  {
    log::info(log::get_ostringstream() << "-priority " << check);
  }
#endif

  /** Check for appearance of -threads, which specifies the maximum number of threads. */
  check = this->GetConfiguration()->GetCommandLineArgument("-threads");
  if (check.empty())
  {
    log::info("-threads  unspecified, so all available threads are used");
  }
  else
  {
    log::info(log::get_ostringstream() << "-threads  " << check);
  }

  /** Check the very important UseDirectionCosines parameter. */
  bool retudc = this->GetConfiguration()->ReadParameter(this->m_UseDirectionCosines, "UseDirectionCosines", 0);
  if (!retudc)
  {
    log::warn(log::get_ostringstream()
              << "\nWARNING: The option \"UseDirectionCosines\" was not found in your parameter file.\n"
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
  this->GetConfiguration()->ReadParameter(randomSeed, "RandomSeed", 0, false);
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
  log::set_precision(this->GetDefaultOutputPrecision());

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
    this->m_MovingImageFileNameContainer =
      GenerateFileNameContainer(*(this->m_Configuration), "-in", inreturndummy, false, true);
    if (inreturndummy != 0)
    {
      log::info("-in       unspecified, so no input image specified");
    }
  }
  /** Check for appearance of "-out". */
  std::string check = this->GetConfiguration()->GetCommandLineArgument("-out");
  if (check.empty())
  {
    log::error(log::get_ostringstream() << "ERROR: No CommandLine option \"-out\" given!");
    returndummy |= 1;
  }
  else
  {
    /** Make sure that last character of -out equals a '/'. */
    if (check.back() != '/')
    {
      this->GetConfiguration()->SetCommandLineArgument("-out", check + '/');
    }
    log::info(log::get_ostringstream() << "-out      " << check);
  }

  /** Check for appearance of -threads, which specifies the maximum number of threads. */
  check = this->GetConfiguration()->GetCommandLineArgument("-threads");
  if (check.empty())
  {
    log::info("-threads  unspecified, so all available threads are used");
  }
  else
  {
    log::info(log::get_ostringstream() << "-threads  " << check);
  }
  if (!BaseComponent::IsElastixLibrary())
  {
    /** Print "-tp". */
    check = this->GetConfiguration()->GetCommandLineArgument("-tp");
    log::info(log::get_ostringstream() << "-tp       " << check);
  }
  /** Check the very important UseDirectionCosines parameter. */
  bool retudc = this->GetConfiguration()->ReadParameter(this->m_UseDirectionCosines, "UseDirectionCosines", 0);
  if (!retudc)
  {
    log::warn(log::get_ostringstream() << "\nWARNING: From elastix 4.3 it is highly recommended to add\n"
                                       << "the UseDirectionCosines option to your parameter file! See\n"
                                       << "http://elastix.lumc.nl/whatsnew_04_3.php for more information.\n");
  }

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
    return this->m_ResultImageContainer->ElementAt(idx).GetPointer();
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
    return this->m_ResultDeformationFieldContainer->ElementAt(idx).GetPointer();
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
  return this->m_UseDirectionCosines;
}


/**
 * ******************** SetOriginalFixedImageDirectionFlat ********************
 */

void
ElastixBase::SetOriginalFixedImageDirectionFlat(const FlatDirectionCosinesType & arg)
{
  this->m_OriginalFixedImageDirection = arg;
}


/**
 * ******************** GetOriginalFixedImageDirectionFlat ********************
 */

const ElastixBase::FlatDirectionCosinesType &
ElastixBase::GetOriginalFixedImageDirectionFlat() const
{
  return this->m_OriginalFixedImageDirection;
}


/**
 * ************** GetTransformParametersMap *****************
 */

itk::ParameterMapInterface::ParameterMapType
ElastixBase::GetTransformParametersMap() const
{
  return this->m_TransformParametersMap;
} // end GetTransformParametersMap()


/**
 * ************** SetConfigurations *********************
 */

void
ElastixBase::SetConfigurations(const std::vector<ConfigurationPointer> & configurations)
{
  this->m_Configurations = configurations;
}


/**
 * ************** GetConfiguration *********************
 */

ElastixBase::ConfigurationPointer
ElastixBase::GetConfiguration(const size_t index) const
{
  return this->m_Configurations[index];
}


} // end namespace elastix
