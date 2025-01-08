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
#ifndef elxTransformBase_hxx
#define elxTransformBase_hxx

#include "elxTransformBase.h"

#include "elxConversion.h"
#include <itkDeref.h>
#include "elxElastixMain.h"
#include "elxTransformIO.h"

#include "itkPointSet.h"
#include "itkDefaultStaticMeshTraits.h"
#include "itkTransformixInputPointFileReader.h"
#include <itksys/SystemTools.hxx>
#include "itkVector.h"
#include "itkTransformToDisplacementFieldFilter.h"
#include "itkTransformToDeterminantOfSpatialJacobianSource.h"
#include "itkTransformToSpatialJacobianSource.h"
#include "itkImageFileWriter.h"
#include "itkImageGridSampler.h"
#include "itkContinuousIndex.h"
#include "itkChangeInformationImageFilter.h"
#include "itkMesh.h"
#include "itkMeshFileReader.h"
#include "itkMeshFileWriter.h"
#include "itkCommonEnums.h"

#include <cassert>
#include <fstream>
#include <iomanip> // For setprecision.


namespace elastix
{


/**
 * ******************** BeforeAllBase ***************************
 */

template <typename TElastix>
int
TransformBase<TElastix>::BeforeAllBase()
{
  const Configuration & configuration = itk::Deref(Superclass::GetConfiguration());

  /** Check Command line options and print them to the logfile. */
  log::info("Command line options from TransformBase:");

  /** Check for appearance of "-t0". */
  if (const std::string commandLineArgument = configuration.GetCommandLineArgument("-t0"); commandLineArgument.empty())
  {
    log::info("-t0       unspecified, so no initial transform used");
  }
  else
  {
    log::info("-t0       " + commandLineArgument);
  }

  /** Return a value. */
  return 0;

} // end BeforeAllBase()


/**
 * ******************** BeforeAllTransformix ********************
 */

template <typename TElastix>
int
TransformBase<TElastix>::BeforeAllTransformix()
{
  const Configuration & configuration = itk::Deref(Superclass::GetConfiguration());

  /** Declare the return value and initialize it. */
  int returndummy = 0;

  /** Check for appearance of "-ipp". */
  if (const std::string commandLineArgument = configuration.GetCommandLineArgument("-ipp");
      !commandLineArgument.empty())
  {
    log::info("-ipp      " + commandLineArgument);
    // Deprecated since elastix 4.3
    log::warn("WARNING: \"-ipp\" is deprecated, use \"-def\" instead!");
  }

  /** Check for appearance of "-def". */
  if (const std::string commandLineArgument = configuration.GetCommandLineArgument("-def"); commandLineArgument.empty())
  {
    log::info("-def      unspecified, so no input points transformed");
  }
  else
  {
    log::info("-def      " + commandLineArgument);
  }

  /** Check for appearance of "-jac". */
  if (const std::string commandLineArgument = configuration.GetCommandLineArgument("-jac"); commandLineArgument.empty())
  {
    log::info("-jac      unspecified, so no det(dT/dx) computed");
  }
  else
  {
    log::info("-jac      " + commandLineArgument);
  }

  /** Check for appearance of "-jacmat". */
  if (const std::string commandLineArgument = configuration.GetCommandLineArgument("-jacmat");
      commandLineArgument.empty())
  {
    log::info("-jacmat   unspecified, so no dT/dx computed");
  }
  else
  {
    log::info("-jacmat   " + commandLineArgument);
  }

  /** Return a value. */
  return returndummy;

} // end BeforeAllTransformix()


/**
 * ******************* BeforeRegistrationBase *******************
 */

template <typename TElastix>
void
TransformBase<TElastix>::BeforeRegistrationBase()
{
  const Configuration & configuration = itk::Deref(Superclass::GetConfiguration());

  /** Read from the configuration file how to combine the initial
   * transform with the current transform.
   */
  std::string howToCombineTransforms = "Compose";
  configuration.ReadParameter(howToCombineTransforms, "HowToCombineTransforms", 0, false);

  this->GetAsITKBaseType()->SetUseComposition(howToCombineTransforms == "Compose");

  /** Set the initial transform. Elastix returns an itk::Object, so try to
   * cast it to an InitialTransformType, which is of type itk::Transform.
   * No need to cast to InitialAdvancedTransformType, since InitialAdvancedTransformType
   * inherits from InitialTransformType.
   */
  if (itk::Object * const object = this->m_Elastix->GetInitialTransform())
  {
    if (auto * const initialTransform = dynamic_cast<InitialTransformType *>(object))
    {
      this->SetInitialTransform(initialTransform);
    }
  }
  else
  {
    std::string fileName = configuration.GetCommandLineArgument("-t0");
    if (fileName.empty())
    {
      const ElastixBase & elastixBase = itk::Deref(Superclass::GetElastix());

      const auto numberOfConfigurations = elastixBase.GetNumberOfTransformConfigurations();

      if ((numberOfConfigurations > 0) && (&configuration != elastixBase.GetTransformConfiguration(0)))
      {
        const Configuration::ConstPointer previousTransformConfiguration =
          elastixBase.GetPreviousTransformConfiguration(configuration);
        const Configuration::ConstPointer lastTransformConfiguration =
          elastixBase.GetTransformConfiguration(numberOfConfigurations - 1);

        this->ReadInitialTransformFromConfiguration(previousTransformConfiguration ? previousTransformConfiguration
                                                                                   : lastTransformConfiguration);
      }
    }
    else
    {
      if (itksys::SystemTools::FileExists(fileName))
      {
        this->ReadInitialTransformFromFile(fileName);
      }
      else
      {
        itkExceptionMacro("ERROR: the file " << fileName << " does not exist!");
      }
    }
  }

} // end BeforeRegistrationBase()


/**
 * ******************* GetInitialTransform **********************
 */

template <typename TElastix>
auto
TransformBase<TElastix>::GetInitialTransform() const -> const InitialTransformType *
{
  return this->GetAsITKBaseType()->GetInitialTransform();

} // end GetInitialTransform()


/**
 * ******************* SetInitialTransform **********************
 */

template <typename TElastix>
void
TransformBase<TElastix>::SetInitialTransform(InitialTransformType * _arg)
{
  /** Set initial transform. */
  this->GetAsITKBaseType()->SetInitialTransform(_arg);

  // \todo AdvancedCombinationTransformType

} // end SetInitialTransform()


/**
 * ******************* SetFinalParameters ********************
 */

template <typename TElastix>
void
TransformBase<TElastix>::SetFinalParameters()
{
  /** Make a local copy, since some transforms do not do this,
   * like the B-spline transform.
   */
  this->m_FinalParameters = this->GetElastix()->GetElxOptimizerBase()->GetAsITKBaseType()->GetCurrentPosition();

  /** Set the final Parameters for the resampler. */
  this->GetAsITKBaseType()->SetParameters(this->m_FinalParameters);

} // end SetFinalParameters()


/**
 * ******************* AfterRegistrationBase ********************
 */

template <typename TElastix>
void
TransformBase<TElastix>::AfterRegistrationBase()
{
  /** Set the final Parameters. */
  this->SetFinalParameters();

} // end AfterRegistrationBase()


/**
 * ******************* ReadFromFile *****************************
 */

template <typename TElastix>
void
TransformBase<TElastix>::ReadFromFile()
{
  /** NOTE:
   * This method assumes the configuration is initialized with a
   * transform parameter file, so not an elastix parameter file!!
   */
  const Configuration & configuration = itk::Deref(Superclass::GetConfiguration());

  /** Task 1 - Read the parameters from file. */

  /** Read the TransformParameters. */
  if (this->m_ReadWriteTransformParameters)
  {
    const auto itkParameterValues = configuration.RetrieveValuesOfParameter<double>("ITKTransformParameters");

    if (itkParameterValues == nullptr)
    {
      /** Get the number of TransformParameters. */
      unsigned int numberOfParameters = 0;
      configuration.ReadParameter(numberOfParameters, "NumberOfParameters", 0);

      /** Read the TransformParameters. */
      std::vector<ValueType> vecPar(numberOfParameters);
      configuration.ReadParameter(vecPar, "TransformParameters", 0, numberOfParameters - 1, true);

      /** Do not rely on vecPar.size(), since it is unchanged by ReadParameter(). */
      const std::size_t numberOfParametersFound = configuration.CountNumberOfParameterEntries("TransformParameters");

      /** Sanity check. Are the number of found parameters the same as
       * the number of specified parameters?
       */
      if (numberOfParametersFound != numberOfParameters)
      {
        itkExceptionMacro("\nERROR: Invalid transform parameter file!\n"
                          << "The number of parameters in \"TransformParameters\" is " << numberOfParametersFound
                          << ", which does not match the number specified in \"NumberOfParameters\" ("
                          << numberOfParameters << ").\n"
                          << "The transform parameters should be specified as:\n"
                          << "  (TransformParameters num num ... num)\n"
                          << "with " << numberOfParameters << " parameters.\n");
      }

      /** Copy to m_TransformParameters. */
      // NOTE: we could avoid this by directly reading into the transform parameters,
      // e.g. by overloading ReadParameter(), or use swap (?).
      m_TransformParameters = Conversion::ToOptimizerParameters(vecPar);
    }
    else
    {
      m_TransformParameters = Conversion::ToOptimizerParameters(*itkParameterValues);

      const auto itkFixedParameterValues =
        configuration.RetrieveValuesOfParameter<double>("ITKTransformFixedParameters");

      if (itkFixedParameterValues != nullptr)
      {
        GetSelf().SetFixedParameters(Conversion::ToOptimizerParameters(*itkFixedParameterValues));
      }
    }

    /** Set the parameters into this transform. */
    this->GetAsITKBaseType()->SetParameters(m_TransformParameters);

  } // end if this->m_ReadWriteTransformParameters

  /** Task 2 - Get the InitialTransform. */
  const ElastixBase & elastixBase = itk::Deref(Superclass::GetElastix());

  if (elastixBase.GetNumberOfTransformConfigurations() > 1)
  {
    const Configuration::ConstPointer previousTransformConfiguration =
      elastixBase.GetPreviousTransformConfiguration(configuration);

    if (previousTransformConfiguration)
    {
      this->ReadInitialTransformFromConfiguration(previousTransformConfiguration);
    }
  }
  else
  {
    /** Get the name of the parameter file that specifies the initial transform. */

    // Retrieve the parameter by its current (preferred) parameter name:
    const auto initialTransformParameterFileName =
      configuration.RetrieveParameterStringValue({}, "InitialTransformParameterFileName", 0, false);
    // Retrieve the parameter by its old (deprecated) parameter name as well:
    const auto initialTransformParametersFileName =
      configuration.RetrieveParameterStringValue({}, "InitialTransformParametersFileName", 0, false);

    if (!initialTransformParametersFileName.empty())
    {
      log::warn("WARNING: The parameter name \"InitialTransformParametersFileName\" is deprecated. Please use "
                "\"InitialTransformParameterFileName\" (without letter 's') instead.");
    }

    // Prefer the value from the current parameter name, otherwise use the old parameter name.
    const auto & fileName = initialTransformParameterFileName.empty() ? initialTransformParametersFileName
                                                                      : initialTransformParameterFileName;

    /** Call the function ReadInitialTransformFromFile. */
    if (!fileName.empty() && fileName != "NoInitialTransform")
    {
      /** Check if the initial transform of this transform parameter file
       * is not the same as this transform parameter file. Otherwise,
       * we will have an infinite loop.
       */

      const std::string configurationParameterFileName = configuration.GetParameterFileName();

      if (itksys::SystemTools::CollapseFullPath(fileName) ==
          itksys::SystemTools::CollapseFullPath(configurationParameterFileName))
      {
        itkExceptionMacro("ERROR: The InitialTransformParameterFileName is identical to the current "
                          "TransformParameters filename! An infinite loop is not allowed.");
      }

      /** We can safely read the initial transform. */

      // Find the last separator (slash or backslash) in the current transform parameter file path.
      const auto lastConfigurationParameterFilePathSeparator = configurationParameterFileName.find_last_of("\\/");
      const char firstFileNameLetter = fileName.front();

      if (const bool isAbsoluteFilePath{ firstFileNameLetter == '\\' || firstFileNameLetter == '/' ||
                                         (firstFileNameLetter > 0 && std::isalpha(firstFileNameLetter) &&
                                          fileName.size() > 1 && fileName[1] == ':') };
          isAbsoluteFilePath || (lastConfigurationParameterFilePathSeparator == std::string::npos) ||
          itksys::SystemTools::FileExists(fileName))
      {
        // The file name is an absolute path, or the current transform parameter file name does not have any separator,
        // or the file exists in the current working directory. So use it!
        this->ReadInitialTransformFromFile(fileName);
      }
      else
      {
        // The file name of the initial transform is a relative path, so now assume that it is relative to the current
        // transform parameter file (the current configuration). Try to read the initial transform from the same
        // directory as the current transform, by concatenating the current configuration file path up to that last
        // separator with the string specified by "InitialTransformParameterFileName".
        this->ReadInitialTransformFromFile(
          configurationParameterFileName.substr(0, lastConfigurationParameterFilePathSeparator + 1) + fileName);
      }
    }
  }

  /** Task 3 - Read from the configuration file how to combine the
   * initial transform with the current transform.
   */
  std::string howToCombineTransforms = "Compose"; // default
  configuration.ReadParameter(howToCombineTransforms, "HowToCombineTransforms", 0, true);

  /** Convert 'this' to a pointer to a CombinationTransform and set how
   * to combine the current transform with the initial transform.
   */
  this->GetAsITKBaseType()->SetUseComposition(howToCombineTransforms == "Compose");

  /** Task 4 - Remember the name of the TransformParameterFileName.
   * This will be needed when another transform will use this transform as an initial transform (see the WriteToFile
   * method), which is relevant for transformix, as well as for elastix (specifically
   * ElastixRegistrationMethod::GenerateData(), when InitialTransformParameterObject is specified).
   */
  this->SetTransformParameterFileName(configuration.GetCommandLineArgument("-tp"));

} // end ReadFromFile()


/**
 * ******************* ReadInitialTransformFromFile *************
 */

template <typename TElastix>
void
TransformBase<TElastix>::ReadInitialTransformFromFile(const std::string & transformParameterFileName)
{
  /** Create a new configuration, which will be initialized with
   * the transformParameterFileName. */
  const auto configurationInitialTransform = Configuration::New();

  if (configurationInitialTransform->Initialize({ { "-tp", transformParameterFileName } }) != 0)
  {
    itkGenericExceptionMacro("ERROR: Reading initial transform parameters failed: " << transformParameterFileName);
  }

  this->ReadInitialTransformFromConfiguration(configurationInitialTransform);

} // end ReadInitialTransformFromFile()


/**
 * ******************* ReadInitialTransformFromConfiguration *****************************
 */

template <typename TElastix>
void
TransformBase<TElastix>::ReadInitialTransformFromConfiguration(
  const Configuration::ConstPointer configurationInitialTransform)
{
  /** Read the InitialTransform name. */
  ComponentDescriptionType initialTransformName = "AffineTransform";
  configurationInitialTransform->ReadParameter(initialTransformName, "Transform", 0);

  /** Create an InitialTransform. */
  if (const PtrToCreator testcreator =
        ElastixMain::GetComponentDatabase().GetCreator(initialTransformName, this->m_Elastix->GetDBIndex()))
  {
    const itk::Object::Pointer initialTransform = testcreator();

    /** Call the ReadFromFile method of the initialTransform. */
    if (const auto elx_initialTransform = dynamic_cast<Self *>(initialTransform.GetPointer()))
    {
      // elx_initialTransform->SetTransformParameterFileName(transformParameterFileName);
      elx_initialTransform->SetElastix(this->GetElastix());
      elx_initialTransform->SetConfiguration(configurationInitialTransform);
      elx_initialTransform->ReadFromFile();

      /** Set initial transform. */
      if (const auto testPointer = dynamic_cast<InitialTransformType *>(initialTransform.GetPointer()))
      {
        this->SetInitialTransform(testPointer);
      }
    }
  }

} // end ReadInitialTransformFromConfiguration()


/**
 * ******************* WriteToFile ******************************
 */

template <typename TElastix>
void
TransformBase<TElastix>::WriteToFile(std::ostream & transformationParameterInfo, const ParametersType & param) const
{
  const Configuration & configuration = itk::Deref(Superclass::GetConfiguration());

  const auto itkTransformOutputFileNameExtensions =
    configuration.GetValuesOfParameter("ITKTransformOutputFileNameExtension");
  const std::string itkTransformOutputFileNameExtension =
    itkTransformOutputFileNameExtensions.empty() ? "" : itkTransformOutputFileNameExtensions.front();

  ParameterMapType parameterMap;

  this->CreateTransformParameterMap(param, parameterMap, itkTransformOutputFileNameExtension.empty());

  const auto & self = GetSelf();

  if (!itkTransformOutputFileNameExtension.empty())
  {
    const auto firstSingleTransform = self.GetNthTransform(0);

    if (firstSingleTransform != nullptr)
    {
      const std::string transformFileName =
        std::string(m_TransformParameterFileName, 0, m_TransformParameterFileName.rfind('.')) + '.' +
        itkTransformOutputFileNameExtension;

      const itk::TransformBase::ConstPointer itkTransform =
        TransformIO::ConvertToSingleItkTransform(*firstSingleTransform);

      TransformIO::Write((itkTransform == nullptr) ? *firstSingleTransform : *itkTransform, transformFileName);

      parameterMap.erase("TransformParameters");
      parameterMap["Transform"] = { "File" };
      parameterMap["TransformFileName"] = { transformFileName };
    }
  }

  const auto writeCompositeTransform = configuration.RetrieveValuesOfParameter<bool>("WriteITKCompositeTransform");

  if ((writeCompositeTransform != nullptr) && (*writeCompositeTransform == std::vector<bool>{ true }) &&
      !itkTransformOutputFileNameExtension.empty())
  {
    const auto compositeTransform = TransformIO::ConvertToItkCompositeTransform(self);

    if (compositeTransform == nullptr)
    {
      log::error(std::ostringstream{}
                 << "Failed to convert a combination of transform to an ITK CompositeTransform. Please check "
                    "that the combination does use composition");
    }
    else
    {
      TransformIO::Write(*compositeTransform,
                         std::string(m_TransformParameterFileName, 0, m_TransformParameterFileName.rfind('.')) +
                           "-Composite." + itkTransformOutputFileNameExtension);
    }
  }

  transformationParameterInfo << Conversion::ParameterMapToString(parameterMap);

  WriteDerivedTransformDataToFile();

} // end WriteToFile()


/**
 * ******************* CreateTransformParameterMap ******************************
 */

template <typename TElastix>
void
TransformBase<TElastix>::CreateTransformParameterMap(const ParametersType & param,
                                                     ParameterMapType &     parameterMap,
                                                     const bool             includeDerivedTransformParameters) const
{
  const Configuration & configuration = itk::Deref(Superclass::GetConfiguration());

  const auto & elastixObject = *(this->GetElastix());

  /** The way Transforms are combined. */
  const auto combinationMethod = this->GetAsITKBaseType()->GetUseAddition() ? "Add" : "Compose";

  /** Write image pixel types. */
  std::string fixpix = "float";
  std::string movpix = "float";

  configuration.ReadParameter(fixpix, "FixedInternalImagePixelType", 0);
  configuration.ReadParameter(movpix, "MovingInternalImagePixelType", 0);

  /** Get the Size, Spacing and Origin of the fixed image. */
  const auto & fixedImage = *(this->m_Elastix->GetFixedImage());
  const auto & largestPossibleRegion = fixedImage.GetLargestPossibleRegion();

  /** The following line would be logically: */
  // auto direction = this->m_Elastix->GetFixedImage()->GetDirection();
  /** But to support the UseDirectionCosines option, we should do it like this: */
  typename FixedImageType::DirectionType direction;
  elastixObject.GetOriginalFixedImageDirection(direction);

  /** Write the name of this transform. */
  parameterMap = { { "Transform", { this->elxGetClassName() } },
                   { "NumberOfParameters", { Conversion::ToString(param.GetSize()) } },
                   { "InitialTransformParameterFileName", { this->GetInitialTransformParameterFileName() } },
                   { "HowToCombineTransforms", { combinationMethod } },
                   { "FixedImageDimension", { Conversion::ToString(FixedImageDimension) } },
                   { "MovingImageDimension", { Conversion::ToString(MovingImageDimension) } },
                   { "FixedInternalImagePixelType", { fixpix } },
                   { "MovingInternalImagePixelType", { movpix } },
                   { "Size", Conversion::ToVectorOfStrings(largestPossibleRegion.GetSize()) },
                   { "Index", Conversion::ToVectorOfStrings(largestPossibleRegion.GetIndex()) },
                   { "Spacing", Conversion::ToVectorOfStrings(fixedImage.GetSpacing()) },
                   { "Origin", Conversion::ToVectorOfStrings(fixedImage.GetOrigin()) },
                   { "Direction", Conversion::ToVectorOfStrings(direction) },
                   { "UseDirectionCosines", { Conversion::ToString(elastixObject.GetUseDirectionCosines()) } } };

  /** Write the parameters of this transform. */
  if (this->m_ReadWriteTransformParameters)
  {
    /** In this case, write in a normal way to the parameter file. */
    parameterMap["TransformParameters"] = { Conversion::ToVectorOfStrings(param) };
  }

  if (includeDerivedTransformParameters)
  {
    // Derived transform classes may add some extra parameters
    for (auto & keyAndValue : this->CreateDerivedTransformParameterMap())
    {
      const auto & key = keyAndValue.first;
      assert(parameterMap.count(key) == 0);
      parameterMap[key] = std::move(keyAndValue.second);
    }
  }

} // end CreateTransformParameterMap()


/**
 * ******************* TransformPoints **************************
 *
 * This function reads points from a file (but only if requested)
 * and transforms these fixed-image coordinates to moving-image
 * coordinates.
 */

template <typename TElastix>
void
TransformBase<TElastix>::TransformPoints() const
{
  const Configuration & configuration = itk::Deref(Superclass::GetConfiguration());

  /** If the optional command "-def" is given in the command
   * line arguments, then and only then we continue.
   */
  const std::string ipp = configuration.GetCommandLineArgument("-ipp");
  std::string       def = configuration.GetCommandLineArgument("-def");

  /** For backwards compatibility def = ipp. */
  if (!def.empty() && !ipp.empty())
  {
    itkExceptionMacro("ERROR: Can not use both \"-def\" and \"-ipp\"!\n"
                      << "  \"-ipp\" is deprecated, use only \"-def\".\n");
  }
  else if (def.empty() && !ipp.empty())
  {
    def = ipp;
  }

  /** If there is an input point-file? */
  if (!def.empty() && def != "all")
  {
    if (itksys::SystemTools::StringEndsWith(def, ".vtk") || itksys::SystemTools::StringEndsWith(def, ".VTK"))
    {
      log::info("  The transform is evaluated on some points, specified in a VTK input point file.");
      this->TransformPointsSomePointsVTK(def);
    }
    else
    {
      log::info("  The transform is evaluated on some points, specified in the input point file.");
      this->TransformPointsSomePoints(def);
    }
  }
  else if (def == "all")
  {
    log::info("  The transform is evaluated on all points. The result is a deformation field.");
    this->TransformPointsAllPoints();
  }
  else
  {
    // just a message
    log::info("  The command-line option \"-def\" is not used, so no points are transformed");
  }

} // end TransformPoints()


/**
 * ************** TransformPointsSomePoints *********************
 *
 * This function reads points from a file and transforms
 * these fixed-image coordinates to moving-image
 * coordinates.
 *
 * Reads the inputpoints from a text file, either as index or as point.
 * Computes the transformed points, converts them back to an index and compute
 * the deformation vector as the difference between the outputpoint and
 * the input point. Save the results.
 */

template <typename TElastix>
void
TransformBase<TElastix>::TransformPointsSomePoints(const std::string & filename) const
{
  /** Typedef's. */
  using FixedImageRegionType = typename FixedImageType::RegionType;
  using FixedImageIndexType = typename FixedImageType::IndexType;
  using FixedImageIndexValueType = typename FixedImageIndexType::IndexValueType;
  using MovingImageIndexType = typename MovingImageType::IndexType;
  using MovingImageIndexValueType = typename MovingImageIndexType::IndexValueType;

  using DummyIPPPixelType = unsigned char;
  using MeshTraitsType =
    itk::DefaultStaticMeshTraits<DummyIPPPixelType, FixedImageDimension, FixedImageDimension, CoordinateType>;
  using PointSetType = itk::PointSet<DummyIPPPixelType, FixedImageDimension, MeshTraitsType>;
  using DeformationVectorType = itk::Vector<float, FixedImageDimension>;

  /** Construct an ipp-file reader. */
  const auto ippReader = itk::TransformixInputPointFileReader<PointSetType>::New();
  ippReader->SetFileName(filename);

  /** Read the input points. */
  log::info(std::ostringstream{} << "  Reading input point file: " << filename);
  try
  {
    ippReader->Update();
  }
  catch (const itk::ExceptionObject & err)
  {
    log::error(std::ostringstream{} << "  Error while opening input point file.\n" << err);
  }

  /** Some user-feedback. */
  if (ippReader->GetPointsAreIndices())
  {
    log::info("  Input points are specified as image indices.");
  }
  else
  {
    log::info("  Input points are specified in world coordinates.");
  }
  const unsigned int nrofpoints = ippReader->GetNumberOfPoints();
  log::info(std::ostringstream{} << "  Number of specified input points: " << nrofpoints);

  /** Get the set of input points. */
  typename PointSetType::Pointer inputPointSet = ippReader->GetOutput();

  /** Create the storage classes. */
  std::vector<FixedImageIndexType>   inputindexvec(nrofpoints);
  std::vector<InputPointType>        inputpointvec(nrofpoints);
  std::vector<OutputPointType>       outputpointvec(nrofpoints);
  std::vector<FixedImageIndexType>   outputindexfixedvec(nrofpoints);
  std::vector<MovingImageIndexType>  outputindexmovingvec(nrofpoints);
  std::vector<DeformationVectorType> deformationvec(nrofpoints);

  const auto & resampleImageFilter = *(this->m_Elastix->GetElxResamplerBase()->GetAsITKBaseType());

  /** Make a temporary image with the right region info,
   * which we can use to convert between points and indices.
   * By taking the image from the resampler output, the UseDirectionCosines
   * parameter is automatically taken into account. */
  const auto dummyImage = FixedImageType::New();
  dummyImage->SetRegions(
    FixedImageRegionType(resampleImageFilter.GetOutputStartIndex(), resampleImageFilter.GetSize()));
  dummyImage->SetOrigin(resampleImageFilter.GetOutputOrigin());
  dummyImage->SetSpacing(resampleImageFilter.GetOutputSpacing());
  dummyImage->SetDirection(resampleImageFilter.GetOutputDirection());

  /** Also output moving image indices if a moving image was supplied. */
  const typename MovingImageType::Pointer movingImage = this->GetElastix()->GetMovingImage();
  const bool                              alsoMovingIndices = movingImage.IsNotNull();

  /** Read the input points, as index or as point. */
  if (ippReader->GetPointsAreIndices())
  {
    for (unsigned int j = 0; j < nrofpoints; ++j)
    {
      /** The read point from the inutPointSet is actually an index
       * Cast to the proper type.
       */
      InputPointType point{};
      inputPointSet->GetPoint(j, &point);
      for (unsigned int i = 0; i < FixedImageDimension; ++i)
      {
        inputindexvec[j][i] = static_cast<FixedImageIndexValueType>(itk::Math::Round<int64_t>(point[i]));
      }
      /** Compute the input point in physical coordinates. */
      dummyImage->TransformIndexToPhysicalPoint(inputindexvec[j], inputpointvec[j]);
    }
  }
  else
  {
    for (unsigned int j = 0; j < nrofpoints; ++j)
    {
      /** Compute index of nearest voxel in fixed image. */
      InputPointType point{};
      inputPointSet->GetPoint(j, &point);
      inputpointvec[j] = point;
      const auto fixedcindex = dummyImage->template TransformPhysicalPointToContinuousIndex<double>(point);
      for (unsigned int i = 0; i < FixedImageDimension; ++i)
      {
        inputindexvec[j][i] = static_cast<FixedImageIndexValueType>(itk::Math::Round<int64_t>(fixedcindex[i]));
      }
    }
  }

  /** Apply the transform. */
  log::info("  The input points are transformed.");
  for (unsigned int j = 0; j < nrofpoints; ++j)
  {
    /** Call TransformPoint. */
    outputpointvec[j] = this->GetAsITKBaseType()->TransformPoint(inputpointvec[j]);

    /** Transform back to index in fixed image domain. */
    const auto fixedcindex = dummyImage->template TransformPhysicalPointToContinuousIndex<double>(outputpointvec[j]);
    for (unsigned int i = 0; i < FixedImageDimension; ++i)
    {
      outputindexfixedvec[j][i] = static_cast<FixedImageIndexValueType>(itk::Math::Round<int64_t>(fixedcindex[i]));
    }

    if (alsoMovingIndices)
    {
      /** Transform back to index in moving image domain. */
      const auto movingcindex =
        movingImage->template TransformPhysicalPointToContinuousIndex<double>(outputpointvec[j]);
      for (unsigned int i = 0; i < MovingImageDimension; ++i)
      {
        outputindexmovingvec[j][i] = static_cast<MovingImageIndexValueType>(itk::Math::Round<int64_t>(movingcindex[i]));
      }
    }

    /** Compute displacement. */
    deformationvec[j].CastFrom(outputpointvec[j] - inputpointvec[j]);
  }

  const Configuration & configuration = itk::Deref(Superclass::GetConfiguration());

  if (const std::string outputDirectoryPath = configuration.GetCommandLineArgument("-out");
      !outputDirectoryPath.empty())
  {
    /** Create filename and file stream. */
    const std::string outputPointsFileName = outputDirectoryPath + "outputpoints.txt";
    std::ofstream     outputPointsFile(outputPointsFileName);
    outputPointsFile << std::showpoint << std::fixed;
    log::info(std::ostringstream{} << "  The transformed points are saved in: " << outputPointsFileName);

    const auto writeToFile = [&outputPointsFile](const auto & rangeOfElements) {
      for (const auto element : rangeOfElements)
      {
        outputPointsFile << element << ' ';
      }
    };

    /** Print the results. */
    for (unsigned int j = 0; j < nrofpoints; ++j)
    {
      /** The input index. */
      outputPointsFile << "Point\t" << j << "\t; InputIndex = [ ";
      writeToFile(inputindexvec[j]);

      /** The input point. */
      outputPointsFile << "]\t; InputPoint = [ ";
      writeToFile(inputpointvec[j]);

      /** The output index in fixed image. */
      outputPointsFile << "]\t; OutputIndexFixed = [ ";
      writeToFile(outputindexfixedvec[j]);

      /** The output point. */
      outputPointsFile << "]\t; OutputPoint = [ ";
      writeToFile(outputpointvec[j]);

      /** The output point minus the input point. */
      outputPointsFile << "]\t; Deformation = [ ";
      writeToFile(deformationvec[j]);

      if (alsoMovingIndices)
      {
        /** The output index in moving image. */
        outputPointsFile << "]\t; OutputIndexMoving = [ ";
        writeToFile(outputindexmovingvec[j]);
      }

      outputPointsFile << "]" << std::endl;
    } // end for nrofpoints
  }

} // end TransformPointsSomePoints()


/**
 * ************** TransformPointsSomePointsVTK *********************
 *
 * This function reads points from a .vtk file and transforms
 * these fixed-image coordinates to moving-image
 * coordinates.
 *
 * Reads the inputmesh from a vtk file, assuming world coordinates.
 * Computes the transformed points, save as outputpoints.vtk.
 */

template <typename TElastix>
void
TransformBase<TElastix>::TransformPointsSomePointsVTK(const std::string & filename) const
{
  const itk::CommonEnums::IOComponent pointComponentType = [&filename] {
    const itk::SmartPointer<itk::MeshIOBase> meshIO =
      itk::MeshIOFactory::CreateMeshIO(filename.c_str(), itk::IOFileModeEnum::ReadMode);
    meshIO->SetFileName(filename);
    meshIO->ReadMeshInformation();
    return meshIO->GetPointComponentType();
  }();

  // Reads a mesh from a VTK file, transforms the mesh, and writes the transformed mesh.
  const auto ReadAndTransformAndWriteMesh = [&filename, this](auto coordinate) {
    // The `coordinate` parameter just specifies the requested coordinate type.
    (void)coordinate;

    using DummyIPPPixelType = unsigned char;
    using MeshType = itk::Mesh<
      DummyIPPPixelType,
      FixedImageDimension,
      itk::DefaultStaticMeshTraits<DummyIPPPixelType, FixedImageDimension, FixedImageDimension, decltype(coordinate)>>;

    /** Read the input points. */
    const auto meshReader = itk::MeshFileReader<MeshType>::New();
    meshReader->SetFileName(filename);
    log::info(std::ostringstream{} << "  Reading input point file: " << filename);
    try
    {
      meshReader->Update();
    }
    catch (const itk::ExceptionObject & err)
    {
      log::error(std::ostringstream{} << "  Error while opening input point file.\n" << err);
    }

    const auto & inputMesh = *(meshReader->GetOutput());

    /** Some user-feedback. */
    log::info("  Input points are specified in world coordinates.");
    const unsigned long nrofpoints = inputMesh.GetNumberOfPoints();
    log::info(std::ostringstream{} << "  Number of specified input points: " << nrofpoints);

    /** Apply the transform. */
    log::info("  The input points are transformed.");

    typename MeshType::ConstPointer outputMesh;

    try
    {
      outputMesh = Self::TransformMesh(inputMesh);
    }
    catch (const itk::ExceptionObject & err)
    {
      log::error(std::ostringstream{} << "  Error while transforming points.\n" << err);
    }

    const Configuration & configuration = itk::Deref(Superclass::GetConfiguration());

    if (const std::string outputDirectoryPath = configuration.GetCommandLineArgument("-out");
        !outputDirectoryPath.empty())
    {
      /** Create filename and file stream. */
      const std::string outputPointsFileName = configuration.GetCommandLineArgument("-out") + "outputpoints.vtk";
      log::info(std::ostringstream{} << "  The transformed points are saved in: " << outputPointsFileName);

      try
      {
        const auto writer = itk::MeshFileWriter<MeshType>::New();
        writer->SetInput(outputMesh);
        writer->SetFileName(outputPointsFileName);

        if (itk::Deref(meshReader->GetModifiableMeshIO()).GetFileType() == itk::IOFileEnum::Binary)
        {
          writer->SetFileTypeAsBINARY();
        }

        writer->Update();
      }
      catch (const itk::ExceptionObject & err)
      {
        log::error(std::ostringstream{} << "  Error while saving points.\n" << err);
      }
    }
  };

  if (pointComponentType == itk::CommonEnums::IOComponent::FLOAT)
  {
    ReadAndTransformAndWriteMesh(float());
  }
  else
  {
    ReadAndTransformAndWriteMesh(double());
  }

} // end TransformPointsSomePointsVTK()


/**
 * ************** TransformPointsAllPoints **********************
 *
 * This function transforms all indexes to a physical point.
 * The difference vector (= the deformation at that index) is
 * stored in an image of vectors (of floats).
 */

template <typename TElastix>
void
TransformBase<TElastix>::TransformPointsAllPoints() const
{
  typename DeformationFieldImageType::Pointer deformationfield = this->GenerateDeformationFieldImage();
  // put deformation field in container
  this->m_Elastix->SetResultDeformationField(deformationfield.GetPointer());

  const Configuration & configuration = itk::Deref(Superclass::GetConfiguration());

  if (!configuration.GetCommandLineArgument("-out").empty())
  {
    WriteDeformationFieldImage(deformationfield);
  }

} // end TransformPointsAllPoints()


/**
 * ************** GenerateDeformationFieldImage **********************
 *
 * This function transforms all indexes to a physical point.
 * The difference vector (= the deformation at that index) is
 * stored in an image of vectors (of floats).
 */

template <typename TElastix>
auto
TransformBase<TElastix>::GenerateDeformationFieldImage() const -> typename DeformationFieldImageType::Pointer
{
  const auto & resampleImageFilter = *(this->m_Elastix->GetElxResamplerBase()->GetAsITKBaseType());

  /** Create an setup deformation field generator. */
  const auto defGenerator = itk::TransformToDisplacementFieldFilter<DeformationFieldImageType, CoordinateType>::New();
  defGenerator->SetSize(resampleImageFilter.GetSize());
  defGenerator->SetOutputSpacing(resampleImageFilter.GetOutputSpacing());
  defGenerator->SetOutputOrigin(resampleImageFilter.GetOutputOrigin());
  defGenerator->SetOutputStartIndex(resampleImageFilter.GetOutputStartIndex());
  defGenerator->SetOutputDirection(resampleImageFilter.GetOutputDirection());
  defGenerator->SetTransform(this->GetAsITKBaseType());

  /** Possibly change direction cosines to their original value, as specified
   * in the tp-file, or by the fixed image. This is only necessary when
   * the UseDirectionCosines flag was set to false. */
  const auto infoChanger = itk::ChangeInformationImageFilter<DeformationFieldImageType>::New();
  typename FixedImageType::DirectionType originalDirection;
  bool                                   retdc = this->GetElastix()->GetOriginalFixedImageDirection(originalDirection);
  infoChanger->SetOutputDirection(originalDirection);
  infoChanger->SetChangeDirection(retdc && !this->GetElastix()->GetUseDirectionCosines());
  infoChanger->SetInput(defGenerator->GetOutput());

  const Configuration & configuration = itk::Deref(Superclass::GetConfiguration());

  /** Track the progress of the generation of the deformation field. */
  const bool showProgressPercentage = configuration.RetrieveParameterValue(false, "ShowProgressPercentage", 0, false);
  const auto progressObserver = showProgressPercentage ? ProgressCommand::CreateAndConnect(*defGenerator) : nullptr;

  try
  {
    infoChanger->Update();
  }
  catch (itk::ExceptionObject & excp)
  {
    /** Add information to the exception. */
    excp.SetLocation("TransformBase - GenerateDeformationFieldImage()");
    std::string err_str = excp.GetDescription();
    err_str += "\nError occurred while generating deformation field image.\n";
    excp.SetDescription(err_str);

    /** Pass the exception to an higher level. */
    throw;
  }

  return infoChanger->GetOutput();
} // end GenerateDeformationFieldImage()


/**
 * ************** WriteDeformationFieldImage **********************
 */

template <typename TElastix>
void
TransformBase<TElastix>::WriteDeformationFieldImage(
  typename TransformBase<TElastix>::DeformationFieldImageType::Pointer deformationfield) const
{
  const Configuration & configuration = itk::Deref(Superclass::GetConfiguration());

  /** Create a name for the deformation field file. */
  std::string resultImageFormat = "mhd";
  configuration.ReadParameter(resultImageFormat, "ResultImageFormat", 0, false);
  std::ostringstream makeFileName;
  makeFileName << configuration.GetCommandLineArgument("-out") << "deformationField." << resultImageFormat;

  /** Write outputImage to disk. */
  log::info("  Computing and writing the deformation field ...");
  try
  {
    itk::WriteImage(deformationfield, makeFileName.str());
  }
  catch (itk::ExceptionObject & excp)
  {
    /** Add information to the exception. */
    excp.SetLocation("TransformBase - WriteDeformationFieldImage()");
    std::string err_str = excp.GetDescription();
    err_str += "\nError occurred while writing deformation field image.\n";
    excp.SetDescription(err_str);

    /** Pass the exception to an higher level. */
    throw;
  }
} // end WriteDeformationFieldImage()


/**
 * ************** ComputeSpatialJacobianDeterminantImage **********************
 */

template <typename TElastix>
auto
TransformBase<TElastix>::ComputeSpatialJacobianDeterminantImage() const ->
  typename SpatialJacobianDeterminantImageType::Pointer
{
  const auto jacGenerator =
    CreateJacobianSource<itk::TransformToDeterminantOfSpatialJacobianSource, SpatialJacobianDeterminantImageType>();
  const auto infoChanger = CreateChangeInformationImageFilter(jacGenerator->GetOutput());
  infoChanger->Update();
  return infoChanger->GetOutput();
}


/**
 * ************** ComputeSpatialJacobianMatrixImage **********************
 */

template <typename TElastix>
auto
TransformBase<TElastix>::ComputeSpatialJacobianMatrixImage() const -> typename SpatialJacobianMatrixImageType::Pointer
{
  const auto jacGenerator =
    CreateJacobianSource<itk::TransformToSpatialJacobianSource, SpatialJacobianMatrixImageType>();
  const auto infoChanger = CreateChangeInformationImageFilter(jacGenerator->GetOutput());
  infoChanger->Update();
  return infoChanger->GetOutput();
}

/**
 * ************** ComputeAndWriteSpatialJacobianDeterminantImage **********************
 */

template <typename TElastix>
void
TransformBase<TElastix>::ComputeAndWriteSpatialJacobianDeterminantImage() const
{
  const Configuration & configuration = itk::Deref(Superclass::GetConfiguration());

  /** If the optional command "-jac" is given in the command line arguments,
   * then and only then we continue.
   */
  std::string jac = configuration.GetCommandLineArgument("-jac");
  if (jac.empty())
  {
    log::info("  The command-line option \"-jac\" is not used, so no det(dT/dx) computed.");
    return;
  }
  else if (jac != "all")
  {
    log::info(std::ostringstream{} << "  WARNING: The command-line option \"-jac\" should be used as \"-jac all\",\n"
                                   << "    but is specified as \"-jac " << jac << "\"\n"
                                   << "    Therefore det(dT/dx) is not computed.");
    return;
  }

  const auto jacGenerator =
    CreateJacobianSource<itk::TransformToDeterminantOfSpatialJacobianSource, SpatialJacobianDeterminantImageType>();
  const auto infoChanger = CreateChangeInformationImageFilter(jacGenerator->GetOutput());

  /** Track the progress of the generation of the deformation field. */
  const bool showProgressPercentage = configuration.RetrieveParameterValue(false, "ShowProgressPercentage", 0, false);
  const auto progressObserver = showProgressPercentage ? ProgressCommand::CreateAndConnect(*jacGenerator) : nullptr;
  /** Create a name for the deformation field file. */
  std::string resultImageFormat = "mhd";
  configuration.ReadParameter(resultImageFormat, "ResultImageFormat", 0, false);

  if (const std::string outputDirectoryPath = configuration.GetCommandLineArgument("-out");
      !outputDirectoryPath.empty())
  {
    std::ostringstream makeFileName;
    makeFileName << outputDirectoryPath << "spatialJacobian." << resultImageFormat;

    /** Write outputImage to disk. */
    log::info("  Computing and writing the spatial Jacobian determinant...");
    try
    {
      itk::WriteImage(infoChanger->GetOutput(), makeFileName.str());
    }
    catch (itk::ExceptionObject & excp)
    {
      /** Add information to the exception. */
      excp.SetLocation("TransformBase - ComputeSpatialJacobianDeterminantImage()");
      std::string err_str = excp.GetDescription();
      err_str += "\nError occurred while writing spatial Jacobian determinant image.\n";
      excp.SetDescription(err_str);

      /** Pass the exception to an higher level. */
      throw;
    }
  }

} // end ComputeAndWriteSpatialJacobianDeterminantImage()


/**
 * ************** ComputeAndWriteSpatialJacobianMatrixImage **********************
 */

template <typename TElastix>
void
TransformBase<TElastix>::ComputeAndWriteSpatialJacobianMatrixImage() const
{
  const Configuration & configuration = itk::Deref(Superclass::GetConfiguration());

  /** If the optional command "-jacmat" is given in the command line arguments,
   * then and only then we continue.
   */
  std::string jac = configuration.GetCommandLineArgument("-jacmat");
  if (jac != "all")
  {
    log::info("  The command-line option \"-jacmat\" is not used, so no dT/dx computed.");
    return;
  }

  const auto jacGenerator =
    CreateJacobianSource<itk::TransformToSpatialJacobianSource, SpatialJacobianMatrixImageType>();

  const auto infoChanger = CreateChangeInformationImageFilter(jacGenerator->GetOutput());

  const bool showProgressPercentage = configuration.RetrieveParameterValue(false, "ShowProgressPercentage", 0, false);
  const auto progressObserver = showProgressPercentage ? ProgressCommand::CreateAndConnect(*jacGenerator) : nullptr;
  /** Create a name for the deformation field file. */
  std::string resultImageFormat = "mhd";
  configuration.ReadParameter(resultImageFormat, "ResultImageFormat", 0, false);

  if (const std::string outputDirectoryPath = configuration.GetCommandLineArgument("-out");
      !outputDirectoryPath.empty())
  {
    std::ostringstream makeFileName;
    makeFileName << outputDirectoryPath << "fullSpatialJacobian." << resultImageFormat;

    /** Write outputImage to disk. */
    const auto jacWriter = itk::ImageFileWriter<SpatialJacobianMatrixImageType>::New();
    jacWriter->SetInput(infoChanger->GetOutput());
    jacWriter->SetFileName(makeFileName.str());

    // This class is used for writing the fullSpatialJacobian image. It is a hack to ensure that a matrix image is seen
    // as a vector image, which most IO classes understand.
    class PixelTypeChangeCommand : public itk::Command
    {
    public:
      ITK_DISALLOW_COPY_AND_MOVE(PixelTypeChangeCommand);

      /** Standard class typedefs. */
      using Self = PixelTypeChangeCommand;
      using Pointer = itk::SmartPointer<Self>;

      /** Run-time type information (and related methods). */
      itkOverrideGetNameOfClassMacro(PixelTypeChangeCommand);

      /** Method for creation through the object factory. */
      itkNewMacro(Self);

    private:
      using PrivateJacobianImageType = SpatialJacobianMatrixImageType;

      /** Set the pixel type to VECTOR */
      void
      Execute(itk::Object * caller, const itk::EventObject &) override
      {
        const auto castcaller = dynamic_cast<itk::ImageFileWriter<PrivateJacobianImageType> *>(caller);
        castcaller->GetModifiableImageIO()->SetPixelType(itk::CommonEnums::IOPixel::VECTOR);
      }

      void
      Execute(const itk::Object * caller, const itk::EventObject & eventObject) override
      {
        // Call the non-const overload.
        Self::Execute(const_cast<itk::Object *>(caller), eventObject);
      }

      PixelTypeChangeCommand() = default;
      ~PixelTypeChangeCommand() override = default;
    };

    /** Hack to change the pixel type to vector. Not necessary for mhd. */
    const auto jacStartWriteCommand = PixelTypeChangeCommand::New();
    if (resultImageFormat != "mhd")
    {
      jacWriter->AddObserver(itk::StartEvent(), jacStartWriteCommand);
    }

    /** Do the writing. */
    log::info("  Computing and writing the spatial Jacobian...");
    try
    {
      jacWriter->Update();
    }
    catch (itk::ExceptionObject & excp)
    {
      /** Add information to the exception. */
      excp.SetLocation("TransformBase - ComputeSpatialJacobianMatrixImage()");
      std::string err_str = excp.GetDescription();
      err_str += "\nError occurred while writing spatial Jacobian image.\n";
      excp.SetDescription(err_str);

      /** Pass the exception to an higher level. */
      throw;
    }
  }

} // end ComputeAndWriteSpatialJacobianMatrixImage()


/**
 * ************** SetTransformParameterFileName ****************
 */

template <typename TElastix>
void
TransformBase<TElastix>::SetTransformParameterFileName(const std::string & filename)
{
  /** Copied from itkSetStringMacro. */
  if (filename == this->m_TransformParameterFileName)
  {
    return;
  }
  this->m_TransformParameterFileName = filename;
  this->GetAsITKBaseType()->Modified();

} // end SetTransformParameterFileName()


/**
 * ************** SetReadWriteTransformParameters ***************
 */

template <typename TElastix>
void
TransformBase<TElastix>::SetReadWriteTransformParameters(const bool _arg)
{
  /** Copied from itkSetMacro. */
  if (this->m_ReadWriteTransformParameters != _arg)
  {
    this->m_ReadWriteTransformParameters = _arg;
    this->GetAsITKBaseType()->Modified();
  }

} // end SetReadWriteTransformParameters()


/**
 * ************** AutomaticScalesEstimation ***************
 */

template <typename TElastix>
void
TransformBase<TElastix>::AutomaticScalesEstimation(ScalesType & scales) const
{
  using ImageSamplerType = itk::ImageGridSampler<FixedImageType>;
  using ImageSampleContainerType = typename ImageSamplerType::ImageSampleContainerType;
  using ImageSampleContainerPointer = typename ImageSampleContainerType::Pointer;

  const ITKBaseType * const thisITK = this->GetAsITKBaseType();
  const unsigned int        outdim = MovingImageDimension;
  const unsigned int        numberOfParameters = thisITK->GetNumberOfParameters();
  scales = ScalesType(numberOfParameters);

  /** Set up grid sampler. */
  const auto sampler = ImageSamplerType::New();
  sampler->SetInput(this->GetRegistration()->GetAsITKBaseType()->GetFixedImage());
  sampler->SetInputImageRegion(this->GetRegistration()->GetAsITKBaseType()->GetFixedImageRegion());

  /** Compute the grid spacing. */
  unsigned long nrofsamples = 10000;
  sampler->SetNumberOfSamples(nrofsamples);

  /** Get samples and check the actually obtained number of samples. */
  sampler->Update();
  ImageSampleContainerPointer sampleContainer = sampler->GetOutput();
  nrofsamples = sampleContainer->Size();
  if (nrofsamples == 0)
  {
    /** \todo: should we demand a minimum number (~100) of voxels? */
    itkExceptionMacro("No valid voxels found to estimate the scales.");
  }

  /** initialize */
  scales.Fill(0.0);

  /** Read fixed coordinates and get Jacobian. */
  for (const auto & sample : *sampleContainer)
  {
    const InputPointType & point = sample.m_ImageCoordinates;
    // const JacobianType & jacobian = thisITK->GetJacobian( point );
    typename ITKBaseType::JacobianType               jacobian;
    typename ITKBaseType::NonZeroJacobianIndicesType nzji;
    thisITK->GetJacobian(point, jacobian, nzji);

    /** Square each element of the Jacobian and add each row
     * to the newscales.
     */
    for (unsigned int d = 0; d < outdim; ++d)
    {
      ScalesType jacd(jacobian[d], numberOfParameters, false);
      scales += element_product(jacd, jacd);
    }
  }
  scales /= static_cast<double>(nrofsamples);

} // end AutomaticScalesEstimation()


/**
 * ************** AutomaticScalesEstimationStackTransform ***************
 */

template <typename TElastix>
void
TransformBase<TElastix>::AutomaticScalesEstimationStackTransform(const unsigned int numberOfSubTransforms,
                                                                 ScalesType &       scales) const
{
  using FixedImageRegionType = typename FixedImageType::RegionType;
  using FixedImageIndexType = typename FixedImageType::IndexType;
  using SizeType = typename FixedImageType::SizeType;

  using ImageSamplerType = itk::ImageGridSampler<FixedImageType>;
  using ImageSampleContainerType = typename ImageSamplerType::ImageSampleContainerType;
  using ImageSampleContainerPointer = typename ImageSampleContainerType::Pointer;

  const ITKBaseType * const thisITK = this->GetAsITKBaseType();
  const unsigned int        outdim = FixedImageDimension;
  const unsigned int        numberOfParameters = thisITK->GetNumberOfParameters();

  /** initialize */
  scales = ScalesType(numberOfParameters);
  scales.Fill(0.0);

  /** Get fixed image region from registration. */
  const FixedImageRegionType & inputRegion = this->GetRegistration()->GetAsITKBaseType()->GetFixedImageRegion();
  SizeType                     size = inputRegion.GetSize();

  /** Set desired extraction region. */
  FixedImageIndexType start = inputRegion.GetIndex();
  start[FixedImageDimension - 1] = size[FixedImageDimension - 1] - 1;

  /** Set size of last dimension to 0. */
  size[FixedImageDimension - 1] = 0;

  log::info(std::ostringstream{} << "start region for scales: " << start << '\n' << "size region for scales: " << size);

  FixedImageRegionType desiredRegion;
  desiredRegion.SetSize(size);
  desiredRegion.SetIndex(start);

  /** Set up the grid sampler. */
  const auto sampler = ImageSamplerType::New();
  sampler->SetInput(this->GetRegistration()->GetAsITKBaseType()->GetFixedImage());
  sampler->SetInputImageRegion(desiredRegion);

  /** Compute the grid spacing. */
  unsigned long nrofsamples = 10000;
  sampler->SetNumberOfSamples(nrofsamples);

  /** Get samples and check the actually obtained number of samples. */
  sampler->Update();
  ImageSampleContainerPointer sampleContainer = sampler->GetOutput();
  nrofsamples = sampleContainer->Size();
  if (nrofsamples == 0)
  {
    /** \todo: should we demand a minimum number (~100) of voxels? */
    itkExceptionMacro("No valid voxels found to estimate the scales.");
  }

  /** Read fixed coordinates and get Jacobian. */
  typename ITKBaseType::JacobianType               jacobian;
  typename ITKBaseType::NonZeroJacobianIndicesType nzji;
  for (const auto & sample : *sampleContainer)
  {
    const InputPointType & point = sample.m_ImageCoordinates;
    // const JacobianType & jacobian = thisITK->GetJacobian( point );
    thisITK->GetJacobian(point, jacobian, nzji);

    /** Square each element of the Jacobian and add each row to the new scales. */
    for (unsigned int d = 0; d < outdim; ++d)
    {
      ScalesType jacd(jacobian[d], numberOfParameters, false);
      scales += element_product(jacd, jacd);
    }
  }
  scales /= static_cast<double>(nrofsamples);

  const unsigned int numberOfScalesSubTransform =
    numberOfParameters / numberOfSubTransforms; //(FixedImageDimension)*(FixedImageDimension - 1);

  for (unsigned int i = 0; i < numberOfParameters; i += numberOfScalesSubTransform)
  {
    for (unsigned int j = 0; j < numberOfScalesSubTransform; ++j)
    {
      scales(i + j) = scales(j);
    }
  }

} // end AutomaticScalesEstimationStackTransform()


} // end namespace elastix

#endif // end #ifndef elxTransformBase_hxx
