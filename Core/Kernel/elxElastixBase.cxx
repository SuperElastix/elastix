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
  std::ostringstream argusedss("");
  argusedss << optionkey << 0;
  std::string argused = argusedss.str();
  std::string check = configuration.GetCommandLineArgument(argused);
  if (check.empty())
  {
    /** Try optionkey. */
    std::ostringstream argusedss2("");
    argusedss2 << optionkey;
    argused = argusedss2.str();
    check = configuration.GetCommandLineArgument(argused);
    if (check.empty())
    {
      /** Both failed; return an error message, if desired. */
      if (printerrors)
      {
        xl::xout["error"] << "ERROR: No CommandLine option \"" << optionkey << "\" or \"" << optionkey << 0
                          << "\" given!" << std::endl;
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
      elxout << argused << spaces << check << std::endl;
    }
    fileNameContainer->CreateElementAt(0) = check;

    /** Loop over all optionkey<i> options given with i > 0. */
    unsigned int i = 1;
    bool         readsuccess = true;
    while (readsuccess)
    {
      std::ostringstream argusedss2("");
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
          elxout << argused << spaces << check << std::endl;
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

ElastixBase::ElastixBase()
{
  /** Initialize. */
  this->m_Configuration = nullptr;
  this->m_DBIndex = 0;

  /** The default output precision of elxout is set to 6. */
  this->m_DefaultOutputPrecision = 6;

  /** Create the component containers. */
  this->m_FixedImagePyramidContainer = ObjectContainerType::New();
  this->m_MovingImagePyramidContainer = ObjectContainerType::New();
  this->m_InterpolatorContainer = ObjectContainerType::New();
  this->m_ImageSamplerContainer = ObjectContainerType::New();
  this->m_MetricContainer = ObjectContainerType::New();
  this->m_OptimizerContainer = ObjectContainerType::New();
  this->m_RegistrationContainer = ObjectContainerType::New();
  this->m_ResamplerContainer = ObjectContainerType::New();
  this->m_ResampleInterpolatorContainer = ObjectContainerType::New();
  this->m_TransformContainer = ObjectContainerType::New();

  /** Create image and mask containers. */
  this->m_FixedImageContainer = DataObjectContainerType::New();
  this->m_MovingImageContainer = DataObjectContainerType::New();
  this->m_FixedImageFileNameContainer = FileNameContainerType::New();
  this->m_MovingImageFileNameContainer = FileNameContainerType::New();

  this->m_FixedMaskContainer = DataObjectContainerType::New();
  this->m_MovingMaskContainer = DataObjectContainerType::New();
  this->m_FixedMaskFileNameContainer = FileNameContainerType::New();
  this->m_MovingMaskFileNameContainer = FileNameContainerType::New();

  this->m_ResultImageContainer = DataObjectContainerType::New();

  /** Initialize initialTransform and final transform. */
  this->m_InitialTransform = nullptr;
  this->m_FinalTransform = nullptr;

  /** From Elastix 4.3 to 4.7: Ignore direction cosines by default, for
   * backward compatability. From Elastix 4.8: set it to true by default.*/
  this->m_UseDirectionCosines = true;

} // end Constructor


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
  elxout << std::setprecision(this->m_DefaultOutputPrecision);

  /** Print to log file. */
  elxout << "ELASTIX version: " ELASTIX_VERSION_STRING "\n";

  /** Check Command line options and print them to the logfile. */
  elxout << "Command line options from ElastixBase:" << std::endl;
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
    elxout << "-fMask    unspecified, so no fixed mask used" << std::endl;
  }
  maskreturndummy = 0;
  this->m_MovingMaskFileNameContainer =
    GenerateFileNameContainer(*(this->m_Configuration), "-mMask", maskreturndummy, false, true);
  if (maskreturndummy != 0)
  {
    elxout << "-mMask    unspecified, so no moving mask used" << std::endl;
  }

  /** Check for appearance of "-out".
   * This check has already been performed in elastix.cxx,
   * Here we do it again. MS: WHY?
   */
  check = this->GetConfiguration()->GetCommandLineArgument("-out");
  if (check.empty())
  {
    xl::xout["error"] << "ERROR: No CommandLine option \"-out\" given!" << std::endl;
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
    elxout << "-out      " << check << std::endl;
  }

  /** Print all "-p". */
  unsigned int i = 1;
  bool         loop = true;
  while (loop)
  {
    std::ostringstream tempPname("");
    tempPname << "-p(" << i << ")";
    check = this->GetConfiguration()->GetCommandLineArgument(tempPname.str());
    if (check.empty())
    {
      loop = false;
    }
    else
    {
      elxout << "-p        " << check << std::endl;
    }
    ++i;
  }

  /** Check for appearance of "-priority", if this is a Windows station. */
#ifdef _WIN32
  check = this->GetConfiguration()->GetCommandLineArgument("-priority");
  if (check.empty())
  {
    elxout << "-priority unspecified, so NORMAL process priority" << std::endl;
  }
  else
  {
    elxout << "-priority " << check << std::endl;
  }
#endif

  /** Check for appearance of -threads, which specifies the maximum number of threads. */
  check = this->GetConfiguration()->GetCommandLineArgument("-threads");
  if (check.empty())
  {
    elxout << "-threads  unspecified, so all available threads are used" << std::endl;
  }
  else
  {
    elxout << "-threads  " << check << std::endl;
  }

  /** Check the very important UseDirectionCosines parameter. */
  bool retudc = this->GetConfiguration()->ReadParameter(this->m_UseDirectionCosines, "UseDirectionCosines", 0);
  if (!retudc)
  {
    xl::xout["warning"] << "\nWARNING: The option \"UseDirectionCosines\" was not found in your parameter file.\n"
                        << "  From elastix 4.8 it defaults to true!\n"
                        << "This may change the behavior of your registrations considerably.\n"
                        << std::endl;
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
  elxout << "ELASTIX version: " ELASTIX_VERSION_STRING "\n";
  elxout << std::setprecision(this->GetDefaultOutputPrecision());

  /** Check Command line options and print them to the logfile. */
  elxout << "Command line options from ElastixBase:" << std::endl;
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
      elxout << "-in       unspecified, so no input image specified" << std::endl;
    }
  }
  /** Check for appearance of "-out". */
  std::string check = this->GetConfiguration()->GetCommandLineArgument("-out");
  if (check.empty())
  {
    xl::xout["error"] << "ERROR: No CommandLine option \"-out\" given!" << std::endl;
    returndummy |= 1;
  }
  else
  {
    /** Make sure that last character of -out equals a '/'. */
    if (check.back() != '/')
    {
      this->GetConfiguration()->SetCommandLineArgument("-out", check + '/');
    }
    elxout << "-out      " << check << std::endl;
  }

  /** Check for appearance of -threads, which specifies the maximum number of threads. */
  check = this->GetConfiguration()->GetCommandLineArgument("-threads");
  if (check.empty())
  {
    elxout << "-threads  unspecified, so all available threads are used" << std::endl;
  }
  else
  {
    elxout << "-threads  " << check << std::endl;
  }
  if (!BaseComponent::IsElastixLibrary())
  {
    /** Print "-tp". */
    check = this->GetConfiguration()->GetCommandLineArgument("-tp");
    elxout << "-tp       " << check << std::endl;
  }
  /** Check the very important UseDirectionCosines parameter. */
  bool retudc = this->GetConfiguration()->ReadParameter(this->m_UseDirectionCosines, "UseDirectionCosines", 0);
  if (!retudc)
  {
    xl::xout["warning"] << "\nWARNING: From elastix 4.3 it is highly recommended to add\n"
                        << "the UseDirectionCosines option to your parameter file! See\n"
                        << "http://elastix.lumc.nl/whatsnew_04_3.php for more information.\n"
                        << std::endl;
  }

  return returndummy;

} // end BeforeAllTransformixBase()


/**
 * ************************ BeforeRegistrationBase ******************
 */

void
ElastixBase::BeforeRegistrationBase()
{
  /** Set up the "iteration" writing field. */
  this->m_IterationInfo.SetOutputs(xl::xout.GetCOutputs());
  this->m_IterationInfo.SetOutputs(xl::xout.GetXOutputs());

} // end BeforeRegistrationBase()


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
