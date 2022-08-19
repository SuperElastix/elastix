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

template <class TElastix>
int
TransformBase<TElastix>::BeforeAllBase()
{
  /** Check Command line options and print them to the logfile. */
  elxout << "Command line options from TransformBase:" << std::endl;
  std::string check("");

  /** Check for appearance of "-t0". */
  check = this->m_Configuration->GetCommandLineArgument("-t0");
  if (check.empty())
  {
    elxout << "-t0       unspecified, so no initial transform used" << std::endl;
  }
  else
  {
    elxout << "-t0       " << check << std::endl;
  }

  /** Return a value. */
  return 0;

} // end BeforeAllBase()


/**
 * ******************** BeforeAllTransformix ********************
 */

template <class TElastix>
int
TransformBase<TElastix>::BeforeAllTransformix()
{
  /** Declare the return value and initialize it. */
  int returndummy = 0;

  /** Declare check. */
  std::string check = "";

  /** Check for appearance of "-ipp". */
  check = this->m_Configuration->GetCommandLineArgument("-ipp");
  if (!check.empty())
  {
    elxout << "-ipp      " << check << std::endl;
    // Deprecated since elastix 4.3
    xl::xout["warning"] << "WARNING: \"-ipp\" is deprecated, use \"-def\" instead!" << std::endl;
  }

  /** Check for appearance of "-def". */
  check = this->m_Configuration->GetCommandLineArgument("-def");
  if (check.empty())
  {
    elxout << "-def      unspecified, so no input points transformed" << std::endl;
  }
  else
  {
    elxout << "-def      " << check << std::endl;
  }

  /** Check for appearance of "-jac". */
  check = this->m_Configuration->GetCommandLineArgument("-jac");
  if (check.empty())
  {
    elxout << "-jac      unspecified, so no det(dT/dx) computed" << std::endl;
  }
  else
  {
    elxout << "-jac      " << check << std::endl;
  }

  /** Check for appearance of "-jacmat". */
  check = this->m_Configuration->GetCommandLineArgument("-jacmat");
  if (check.empty())
  {
    elxout << "-jacmat   unspecified, so no dT/dx computed" << std::endl;
  }
  else
  {
    elxout << "-jacmat   " << check << std::endl;
  }

  /** Return a value. */
  return returndummy;

} // end BeforeAllTransformix()


/**
 * ******************* BeforeRegistrationBase *******************
 */

template <class TElastix>
void
TransformBase<TElastix>::BeforeRegistrationBase()
{
  /** Read from the configuration file how to combine the initial
   * transform with the current transform.
   */
  std::string howToCombineTransforms = "Compose";
  this->m_Configuration->ReadParameter(howToCombineTransforms, "HowToCombineTransforms", 0, false);

  this->GetAsITKBaseType()->SetUseComposition(howToCombineTransforms == "Compose");

  /** Set the initial transform. Elastix returns an itk::Object, so try to
   * cast it to an InitialTransformType, which is of type itk::Transform.
   * No need to cast to InitialAdvancedTransformType, since InitialAdvancedTransformType
   * inherits from InitialTransformType.
   */
  if (this->m_Elastix->GetInitialTransform())
  {
    InitialTransformType * testPointer = dynamic_cast<InitialTransformType *>(this->m_Elastix->GetInitialTransform());
    if (testPointer)
    {
      this->SetInitialTransform(testPointer);
    }
  }
  else
  {
    std::string fileName = this->m_Configuration->GetCommandLineArgument("-t0");
    if (!fileName.empty())
    {
      if (itksys::SystemTools::FileExists(fileName.c_str()))
      {
        this->ReadInitialTransformFromFile(fileName.c_str());
      }
      else
      {
        itkExceptionMacro(<< "ERROR: the file " << fileName << " does not exist!");
      }
    }
  }

} // end BeforeRegistrationBase()


/**
 * ******************* GetInitialTransform **********************
 */

template <class TElastix>
auto
TransformBase<TElastix>::GetInitialTransform() const -> const InitialTransformType *
{
  return this->GetAsITKBaseType()->GetInitialTransform();

} // end GetInitialTransform()


/**
 * ******************* SetInitialTransform **********************
 */

template <class TElastix>
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

template <class TElastix>
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

template <class TElastix>
void
TransformBase<TElastix>::AfterRegistrationBase()
{
  /** Set the final Parameters. */
  this->SetFinalParameters();

} // end AfterRegistrationBase()


/**
 * ******************* ReadFromFile *****************************
 */

template <class TElastix>
void
TransformBase<TElastix>::ReadFromFile()
{
  /** NOTE:
   * This method assumes this->m_Configuration is initialized with a
   * transform parameter file, so not an elastix parameter file!!
   */

  /** Task 1 - Read the parameters from file. */

  /** Read the TransformParameters. */
  if (this->m_ReadWriteTransformParameters)
  {
    const auto itkParameterValues =
      this->m_Configuration->template RetrieveValuesOfParameter<double>("ITKTransformParameters");

    if (itkParameterValues == nullptr)
    {
      /** Get the number of TransformParameters. */
      unsigned int numberOfParameters = 0;
      this->m_Configuration->ReadParameter(numberOfParameters, "NumberOfParameters", 0);

      /** Read the TransformParameters. */
      std::vector<ValueType> vecPar(numberOfParameters);
      this->m_Configuration->ReadParameter(vecPar, "TransformParameters", 0, numberOfParameters - 1, true);

      /** Do not rely on vecPar.size(), since it is unchanged by ReadParameter(). */
      const std::size_t numberOfParametersFound =
        this->m_Configuration->CountNumberOfParameterEntries("TransformParameters");

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
        this->m_Configuration->template RetrieveValuesOfParameter<double>("ITKTransformFixedParameters");

      if (itkFixedParameterValues != nullptr)
      {
        GetSelf().SetFixedParameters(Conversion::ToOptimizerParameters(*itkFixedParameterValues));
      }
    }

    /** Set the parameters into this transform. */
    this->GetAsITKBaseType()->SetParameters(m_TransformParameters);

  } // end if this->m_ReadWriteTransformParameters

  /** Task 2 - Get the InitialTransform. */

  /** Get the InitialTransformName. */
  std::string fileName = "NoInitialTransform";
  this->m_Configuration->ReadParameter(fileName, "InitialTransformParametersFileName", 0);

  /** Call the function ReadInitialTransformFromFile. */
  if (fileName != "NoInitialTransform")
  {
    // The value of "InitialTransformParametersFileName" is either an index
    // (size_t) into the vector of configuration objects, or the file name of
    // a transform parameters file. If it is an index, call
    // ReadInitialTransformFromConfiguration, otherwise call ReadInitialTransformFromFile.

    /** Get initial transform index number. */
    std::istringstream to_size_t(fileName);
    size_t             index;
    to_size_t >> index;

    if (to_size_t.eof() && !to_size_t.fail())
    {
      /** We can safely read the initial transform. */
      // Retrieve configuration object from internally stored vector of configuration objects.
      this->ReadInitialTransformFromConfiguration(this->GetElastix()->GetConfiguration(index));
    }
    else
    {
      /** Check if the initial transform of this transform parameter file
       * is not the same as this transform parameter file. Otherwise,
       * we will have an infinite loop.
       */
      std::string fullFileName1 = itksys::SystemTools::CollapseFullPath(fileName);
      std::string fullFileName2 = itksys::SystemTools::CollapseFullPath(this->m_Configuration->GetParameterFileName());
      if (fullFileName1 == fullFileName2)
      {
        itkExceptionMacro(<< "ERROR: The InitialTransformParametersFileName is identical to the current "
                             "TransformParameters filename! An infinite loop is not allowed.");
      }

      /** We can safely read the initial transform. */
      this->ReadInitialTransformFromFile(fileName.c_str());
    }
  }

  /** Task 3 - Read from the configuration file how to combine the
   * initial transform with the current transform.
   */
  std::string howToCombineTransforms = "Compose"; // default
  this->m_Configuration->ReadParameter(howToCombineTransforms, "HowToCombineTransforms", 0, true);

  /** Convert 'this' to a pointer to a CombinationTransform and set how
   * to combine the current transform with the initial transform.
   */
  this->GetAsITKBaseType()->SetUseComposition(howToCombineTransforms == "Compose");

  /** Task 4 - Remember the name of the TransformParametersFileName.
   * This will be needed when another transform will use this transform
   * as an initial transform (see the WriteToFile method)
   */
  this->SetTransformParametersFileName(this->GetConfiguration()->GetCommandLineArgument("-tp").c_str());

} // end ReadFromFile()


/**
 * ******************* ReadInitialTransformFromFile *************
 */

template <class TElastix>
void
TransformBase<TElastix>::ReadInitialTransformFromFile(const char * transformParametersFileName)
{
  /** Create a new configuration, which will be initialized with
   * the transformParameterFileName. */
  const auto configurationInitialTransform = Configuration::New();

  if (configurationInitialTransform->Initialize({ { "-tp", transformParametersFileName } }) != 0)
  {
    itkGenericExceptionMacro(<< "ERROR: Reading initial transform parameters failed: " << transformParametersFileName);
  }

  this->ReadInitialTransformFromConfiguration(configurationInitialTransform);

} // end ReadInitialTransformFromFile()


/**
 * ******************* ReadInitialTransformFromConfiguration *****************************
 */

template <class TElastix>
void
TransformBase<TElastix>::ReadInitialTransformFromConfiguration(
  const Configuration::Pointer configurationInitialTransform)
{
  /** Read the InitialTransform name. */
  ComponentDescriptionType initialTransformName = "AffineTransform";
  configurationInitialTransform->ReadParameter(initialTransformName, "Transform", 0);

  /** Create an InitialTransform. */
  const PtrToCreator testcreator =
    ElastixMain::GetComponentDatabase().GetCreator(initialTransformName, this->m_Elastix->GetDBIndex());
  const itk::Object::Pointer initialTransform = (testcreator == nullptr) ? nullptr : testcreator();

  const auto elx_initialTransform = dynamic_cast<Self *>(initialTransform.GetPointer());

  /** Call the ReadFromFile method of the initialTransform. */
  if (elx_initialTransform != nullptr)
  {
    // elx_initialTransform->SetTransformParametersFileName(transformParametersFileName);
    elx_initialTransform->SetElastix(this->GetElastix());
    elx_initialTransform->SetConfiguration(configurationInitialTransform);
    elx_initialTransform->ReadFromFile();

    /** Set initial transform. */
    const auto testPointer = dynamic_cast<InitialTransformType *>(initialTransform.GetPointer());
    if (testPointer != nullptr)
    {
      this->SetInitialTransform(testPointer);
    }
  }

} // end ReadInitialTransformFromConfiguration()


/**
 * ******************* WriteToFile ******************************
 */

template <class TElastix>
void
TransformBase<TElastix>::WriteToFile(xl::xoutsimple & transformationParameterInfo, const ParametersType & param) const
{
  const auto & configuration = *(this->Superclass::m_Configuration);
  const auto   itkTransformOutputFileNameExtensions =
    configuration.GetValuesOfParameter("ITKTransformOutputFileNameExtension");
  const std::string itkTransformOutputFileNameExtension =
    itkTransformOutputFileNameExtensions.empty() ? "" : itkTransformOutputFileNameExtensions.front();

  ParameterMapType parameterMap;

  this->CreateTransformParametersMap(param, parameterMap, itkTransformOutputFileNameExtension.empty());

  const auto & self = GetSelf();

  if (!itkTransformOutputFileNameExtension.empty())
  {
    const auto firstSingleTransform = self.GetNthTransform(0);

    if (firstSingleTransform != nullptr)
    {
      const std::string transformFileName =
        std::string(m_TransformParametersFileName, 0, m_TransformParametersFileName.rfind('.')) + '.' +
        itkTransformOutputFileNameExtension;

      const itk::TransformBase::ConstPointer itkTransform =
        TransformIO::ConvertToSingleItkTransform(*firstSingleTransform);

      TransformIO::Write((itkTransform == nullptr) ? *firstSingleTransform : *itkTransform, transformFileName);

      parameterMap.erase("TransformParameters");
      parameterMap["Transform"] = { "File" };
      parameterMap["TransformFileName"] = { transformFileName };
    }
  }

  const auto writeCompositeTransform =
    configuration.template RetrieveValuesOfParameter<bool>("WriteITKCompositeTransform");

  if ((writeCompositeTransform != nullptr) && (*writeCompositeTransform == std::vector<bool>{ true }) &&
      !itkTransformOutputFileNameExtension.empty())
  {
    const auto compositeTransform = TransformIO::ConvertToItkCompositeTransform(self);

    if (compositeTransform == nullptr)
    {
      xl::xout["error"] << "Failed to convert a combination of transform to an ITK CompositeTransform. Please check "
                           "that the combination does use composition"
                        << std::endl;
    }
    else
    {
      TransformIO::Write(*compositeTransform,
                         std::string(m_TransformParametersFileName, 0, m_TransformParametersFileName.rfind('.')) +
                           "-Composite." + itkTransformOutputFileNameExtension);
    }
  }

  transformationParameterInfo << Conversion::ParameterMapToString(parameterMap);

  WriteDerivedTransformDataToFile();

} // end WriteToFile()


/**
 * ******************* CreateTransformParametersMap ******************************
 */

template <class TElastix>
void
TransformBase<TElastix>::CreateTransformParametersMap(const ParametersType & param,
                                                      ParameterMapType &     parameterMap,
                                                      const bool             includeDerivedTransformParameters) const
{
  const auto & elastixObject = *(this->GetElastix());

  /** The way Transforms are combined. */
  const auto combinationMethod = this->GetAsITKBaseType()->GetUseAddition() ? "Add" : "Compose";

  /** Write image pixel types. */
  std::string fixpix = "float";
  std::string movpix = "float";

  this->m_Configuration->ReadParameter(fixpix, "FixedInternalImagePixelType", 0);
  this->m_Configuration->ReadParameter(movpix, "MovingInternalImagePixelType", 0);

  /** Get the Size, Spacing and Origin of the fixed image. */
  const auto & fixedImage = *(this->m_Elastix->GetFixedImage());
  const auto & largestPossibleRegion = fixedImage.GetLargestPossibleRegion();

  /** The following line would be logically: */
  // FixedImageDirectionType direction =
  //  this->m_Elastix->GetFixedImage()->GetDirection();
  /** But to support the UseDirectionCosines option, we should do it like this: */
  typename FixedImageType::DirectionType direction;
  elastixObject.GetOriginalFixedImageDirection(direction);

  /** Write the name of this transform. */
  parameterMap = { { "Transform", { this->elxGetClassName() } },
                   { "NumberOfParameters", { Conversion::ToString(param.GetSize()) } },
                   { "InitialTransformParametersFileName", { this->GetInitialTransformParametersFileName() } },
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
    for (auto & keyAndValue : this->CreateDerivedTransformParametersMap())
    {
      const auto & key = keyAndValue.first;
      assert(parameterMap.count(key) == 0);
      parameterMap[key] = std::move(keyAndValue.second);
    }
  }

} // end CreateTransformParametersMap()


/**
 * ******************* TransformPoints **************************
 *
 * This function reads points from a file (but only if requested)
 * and transforms these fixed-image coordinates to moving-image
 * coordinates.
 */

template <class TElastix>
void
TransformBase<TElastix>::TransformPoints() const
{
  /** If the optional command "-def" is given in the command
   * line arguments, then and only then we continue.
   */
  const std::string ipp = this->GetConfiguration()->GetCommandLineArgument("-ipp");
  std::string       def = this->GetConfiguration()->GetCommandLineArgument("-def");

  /** For backwards compatibility def = ipp. */
  if (!def.empty() && !ipp.empty())
  {
    itkExceptionMacro(<< "ERROR: Can not use both \"-def\" and \"-ipp\"!\n"
                      << "  \"-ipp\" is deprecated, use only \"-def\".\n");
  }
  else if (def.empty() && !ipp.empty())
  {
    def = ipp;
  }

  /** If there is an input point-file? */
  if (!def.empty() && def != "all")
  {
    if (itksys::SystemTools::StringEndsWith(def.c_str(), ".vtk") ||
        itksys::SystemTools::StringEndsWith(def.c_str(), ".VTK"))
    {
      elxout << "  The transform is evaluated on some points, specified in a VTK input point file." << std::endl;
      this->TransformPointsSomePointsVTK(def);
    }
    else
    {
      elxout << "  The transform is evaluated on some points, specified in the input point file." << std::endl;
      this->TransformPointsSomePoints(def);
    }
  }
  else if (def == "all")
  {
    elxout << "  The transform is evaluated on all points. The result is a deformation field." << std::endl;
    this->TransformPointsAllPoints();
  }
  else
  {
    // just a message
    elxout << "  The command-line option \"-def\" is not used, so no points are transformed" << std::endl;
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

template <class TElastix>
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
    itk::DefaultStaticMeshTraits<DummyIPPPixelType, FixedImageDimension, FixedImageDimension, CoordRepType>;
  using PointSetType = itk::PointSet<DummyIPPPixelType, FixedImageDimension, MeshTraitsType>;
  using DeformationVectorType = itk::Vector<float, FixedImageDimension>;

  /** Construct an ipp-file reader. */
  const auto ippReader = itk::TransformixInputPointFileReader<PointSetType>::New();
  ippReader->SetFileName(filename.c_str());

  /** Read the input points. */
  elxout << "  Reading input point file: " << filename << std::endl;
  try
  {
    ippReader->Update();
  }
  catch (const itk::ExceptionObject & err)
  {
    xl::xout["error"] << "  Error while opening input point file." << std::endl;
    xl::xout["error"] << err << std::endl;
  }

  /** Some user-feedback. */
  if (ippReader->GetPointsAreIndices())
  {
    elxout << "  Input points are specified as image indices." << std::endl;
  }
  else
  {
    elxout << "  Input points are specified in world coordinates." << std::endl;
  }
  const unsigned int nrofpoints = ippReader->GetNumberOfPoints();
  elxout << "  Number of specified input points: " << nrofpoints << std::endl;

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
        inputindexvec[j][i] = static_cast<FixedImageIndexValueType>(itk::Math::Round<double>(point[i]));
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
        inputindexvec[j][i] = static_cast<FixedImageIndexValueType>(itk::Math::Round<double>(fixedcindex[i]));
      }
    }
  }

  /** Apply the transform. */
  elxout << "  The input points are transformed." << std::endl;
  for (unsigned int j = 0; j < nrofpoints; ++j)
  {
    /** Call TransformPoint. */
    outputpointvec[j] = this->GetAsITKBaseType()->TransformPoint(inputpointvec[j]);

    /** Transform back to index in fixed image domain. */
    const auto fixedcindex = dummyImage->template TransformPhysicalPointToContinuousIndex<double>(outputpointvec[j]);
    for (unsigned int i = 0; i < FixedImageDimension; ++i)
    {
      outputindexfixedvec[j][i] = static_cast<FixedImageIndexValueType>(itk::Math::Round<double>(fixedcindex[i]));
    }

    if (alsoMovingIndices)
    {
      /** Transform back to index in moving image domain. */
      const auto movingcindex =
        movingImage->template TransformPhysicalPointToContinuousIndex<double>(outputpointvec[j]);
      for (unsigned int i = 0; i < MovingImageDimension; ++i)
      {
        outputindexmovingvec[j][i] = static_cast<MovingImageIndexValueType>(itk::Math::Round<double>(movingcindex[i]));
      }
    }

    /** Compute displacement. */
    deformationvec[j].CastFrom(outputpointvec[j] - inputpointvec[j]);
  }

  /** Create filename and file stream. */
  const std::string outputPointsFileName = this->m_Configuration->GetCommandLineArgument("-out") + "outputpoints.txt";
  std::ofstream     outputPointsFile(outputPointsFileName);
  outputPointsFile << std::showpoint << std::fixed;
  elxout << "  The transformed points are saved in: " << outputPointsFileName << std::endl;

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

template <class TElastix>
void
TransformBase<TElastix>::TransformPointsSomePointsVTK(const std::string & filename) const
{
  /** Typedef's. \todo test DummyIPPPixelType=bool. */
  using DummyIPPPixelType = float;
  using MeshTraitsType =
    itk::DefaultStaticMeshTraits<DummyIPPPixelType, FixedImageDimension, FixedImageDimension, CoordRepType>;
  using MeshType = itk::Mesh<DummyIPPPixelType, FixedImageDimension, MeshTraitsType>;

  /** Read the input points. */
  const auto meshReader = itk::MeshFileReader<MeshType>::New();
  meshReader->SetFileName(filename.c_str());
  elxout << "  Reading input point file: " << filename << std::endl;
  try
  {
    meshReader->Update();
  }
  catch (const itk::ExceptionObject & err)
  {
    xl::xout["error"] << "  Error while opening input point file." << std::endl;
    xl::xout["error"] << err << std::endl;
  }

  const auto & inputMesh = *(meshReader->GetOutput());

  /** Some user-feedback. */
  elxout << "  Input points are specified in world coordinates." << std::endl;
  const unsigned long nrofpoints = inputMesh.GetNumberOfPoints();
  elxout << "  Number of specified input points: " << nrofpoints << std::endl;

  /** Apply the transform. */
  elxout << "  The input points are transformed." << std::endl;

  typename MeshType::ConstPointer outputMesh;

  try
  {
    outputMesh = Self::TransformMesh(inputMesh);
  }
  catch (const itk::ExceptionObject & err)
  {
    xl::xout["error"] << "  Error while transforming points." << std::endl;
    xl::xout["error"] << err << std::endl;
  }

  /** Create filename and file stream. */
  const std::string outputPointsFileName = this->m_Configuration->GetCommandLineArgument("-out") + "outputpoints.vtk";
  elxout << "  The transformed points are saved in: " << outputPointsFileName << std::endl;

  try
  {
    itk::WriteMesh(outputMesh, outputPointsFileName);
  }
  catch (const itk::ExceptionObject & err)
  {
    xl::xout["error"] << "  Error while saving points." << std::endl;
    xl::xout["error"] << err << std::endl;
  }

} // end TransformPointsSomePointsVTK()


/**
 * ************** TransformPointsAllPoints **********************
 *
 * This function transforms all indexes to a physical point.
 * The difference vector (= the deformation at that index) is
 * stored in an image of vectors (of floats).
 */

template <class TElastix>
void
TransformBase<TElastix>::TransformPointsAllPoints() const
{
  typename DeformationFieldImageType::Pointer deformationfield = this->GenerateDeformationFieldImage();
  // put deformation field in container
  this->m_Elastix->SetResultDeformationField(deformationfield.GetPointer());

  if (!BaseComponent::IsElastixLibrary())
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

template <class TElastix>
auto
TransformBase<TElastix>::GenerateDeformationFieldImage() const -> typename DeformationFieldImageType::Pointer
{
  /** Typedef's. */
  using FixedImageDirectionType = typename FixedImageType::DirectionType;
  using DeformationFieldGeneratorType =
    itk::TransformToDisplacementFieldFilter<DeformationFieldImageType, CoordRepType>;
  using ChangeInfoFilterType = itk::ChangeInformationImageFilter<DeformationFieldImageType>;

  const auto & resampleImageFilter = *(this->m_Elastix->GetElxResamplerBase()->GetAsITKBaseType());

  /** Create an setup deformation field generator. */
  const auto defGenerator = DeformationFieldGeneratorType::New();
  defGenerator->SetSize(resampleImageFilter.GetSize());
  defGenerator->SetOutputSpacing(resampleImageFilter.GetOutputSpacing());
  defGenerator->SetOutputOrigin(resampleImageFilter.GetOutputOrigin());
  defGenerator->SetOutputStartIndex(resampleImageFilter.GetOutputStartIndex());
  defGenerator->SetOutputDirection(resampleImageFilter.GetOutputDirection());
  defGenerator->SetTransform(const_cast<const ITKBaseType *>(this->GetAsITKBaseType()));

  /** Possibly change direction cosines to their original value, as specified
   * in the tp-file, or by the fixed image. This is only necessary when
   * the UseDirectionCosines flag was set to false. */
  const auto              infoChanger = ChangeInfoFilterType::New();
  FixedImageDirectionType originalDirection;
  bool                    retdc = this->GetElastix()->GetOriginalFixedImageDirection(originalDirection);
  infoChanger->SetOutputDirection(originalDirection);
  infoChanger->SetChangeDirection(retdc & !this->GetElastix()->GetUseDirectionCosines());
  infoChanger->SetInput(defGenerator->GetOutput());

  /** Track the progress of the generation of the deformation field. */
  const auto progressObserver =
    BaseComponent::IsElastixLibrary() ? nullptr : ProgressCommandType::CreateAndConnect(*defGenerator);

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

template <class TElastix>
void
TransformBase<TElastix>::WriteDeformationFieldImage(
  typename TransformBase<TElastix>::DeformationFieldImageType::Pointer deformationfield) const
{
  /** Create a name for the deformation field file. */
  std::string resultImageFormat = "mhd";
  this->m_Configuration->ReadParameter(resultImageFormat, "ResultImageFormat", 0, false);
  std::ostringstream makeFileName;
  makeFileName << this->m_Configuration->GetCommandLineArgument("-out") << "deformationField." << resultImageFormat;

  /** Write outputImage to disk. */
  elxout << "  Computing and writing the deformation field ..." << std::endl;
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
 * ************** ComputeDeterminantOfSpatialJacobian **********************
 */

template <class TElastix>
void
TransformBase<TElastix>::ComputeDeterminantOfSpatialJacobian() const
{
  /** If the optional command "-jac" is given in the command line arguments,
   * then and only then we continue.
   */
  std::string jac = this->GetConfiguration()->GetCommandLineArgument("-jac");
  if (jac.empty())
  {
    elxout << "  The command-line option \"-jac\" is not used, so no det(dT/dx) computed." << std::endl;
    return;
  }
  else if (jac != "all")
  {
    elxout << "  WARNING: The command-line option \"-jac\" should be used as \"-jac all\",\n"
           << "    but is specified as \"-jac " << jac << "\"\n"
           << "    Therefore det(dT/dx) is not computed." << std::endl;
    return;
  }

  /** Typedef's. */
  using JacobianImageType = itk::Image<float, FixedImageDimension>;
  using JacobianGeneratorType = itk::TransformToDeterminantOfSpatialJacobianSource<JacobianImageType, CoordRepType>;
  using ChangeInfoFilterType = itk::ChangeInformationImageFilter<JacobianImageType>;
  using FixedImageDirectionType = typename FixedImageType::DirectionType;

  const auto & resampleImageFilter = *(this->m_Elastix->GetElxResamplerBase()->GetAsITKBaseType());

  /** Create an setup Jacobian generator. */
  const auto jacGenerator = JacobianGeneratorType::New();
  jacGenerator->SetTransform(const_cast<const ITKBaseType *>(this->GetAsITKBaseType()));
  jacGenerator->SetOutputSize(resampleImageFilter.GetSize());
  jacGenerator->SetOutputSpacing(resampleImageFilter.GetOutputSpacing());
  jacGenerator->SetOutputOrigin(resampleImageFilter.GetOutputOrigin());
  jacGenerator->SetOutputIndex(resampleImageFilter.GetOutputStartIndex());
  jacGenerator->SetOutputDirection(resampleImageFilter.GetOutputDirection());
  // NOTE: We can not use the following, since the fixed image does not exist in transformix
  //   jacGenerator->SetOutputParametersFromImage(
  //     this->GetRegistration()->GetAsITKBaseType()->GetFixedImage() );

  /** Possibly change direction cosines to their original value, as specified
   * in the tp-file, or by the fixed image. This is only necessary when
   * the UseDirectionCosines flag was set to false. */
  const auto              infoChanger = ChangeInfoFilterType::New();
  FixedImageDirectionType originalDirection;
  bool                    retdc = this->GetElastix()->GetOriginalFixedImageDirection(originalDirection);
  infoChanger->SetOutputDirection(originalDirection);
  infoChanger->SetChangeDirection(retdc & !this->GetElastix()->GetUseDirectionCosines());
  infoChanger->SetInput(jacGenerator->GetOutput());

  /** Track the progress of the generation of the deformation field. */
  const auto progressObserver =
    BaseComponent::IsElastixLibrary() ? nullptr : ProgressCommandType::CreateAndConnect(*jacGenerator);
  /** Create a name for the deformation field file. */
  std::string resultImageFormat = "mhd";
  this->m_Configuration->ReadParameter(resultImageFormat, "ResultImageFormat", 0, false);
  std::ostringstream makeFileName;
  makeFileName << this->m_Configuration->GetCommandLineArgument("-out") << "spatialJacobian." << resultImageFormat;

  /** Write outputImage to disk. */
  elxout << "  Computing and writing the spatial Jacobian determinant..." << std::endl;
  try
  {
    itk::WriteImage(infoChanger->GetOutput(), makeFileName.str());
  }
  catch (itk::ExceptionObject & excp)
  {
    /** Add information to the exception. */
    excp.SetLocation("TransformBase - ComputeDeterminantOfSpatialJacobian()");
    std::string err_str = excp.GetDescription();
    err_str += "\nError occurred while writing spatial Jacobian determinant image.\n";
    excp.SetDescription(err_str);

    /** Pass the exception to an higher level. */
    throw;
  }

} // end ComputeDeterminantOfSpatialJacobian()


/**
 * ************** ComputeSpatialJacobian **********************
 */

template <class TElastix>
void
TransformBase<TElastix>::ComputeSpatialJacobian() const
{
  /** If the optional command "-jacmat" is given in the command line arguments,
   * then and only then we continue.
   */
  std::string jac = this->GetConfiguration()->GetCommandLineArgument("-jacmat");
  if (jac != "all")
  {
    elxout << "  The command-line option \"-jacmat\" is not used, so no dT/dx computed." << std::endl;
    return;
  }

  /** Typedef's. */
  using SpatialJacobianComponentType = float;
  using OutputSpatialJacobianType =
    itk::Matrix<SpatialJacobianComponentType, MovingImageDimension, FixedImageDimension>;
  using JacobianImageType = itk::Image<OutputSpatialJacobianType, FixedImageDimension>;
  using JacobianGeneratorType = itk::TransformToSpatialJacobianSource<JacobianImageType, CoordRepType>;
  using ChangeInfoFilterType = itk::ChangeInformationImageFilter<JacobianImageType>;
  using FixedImageDirectionType = typename FixedImageType::DirectionType;

  const auto & resampleImageFilter = *(this->m_Elastix->GetElxResamplerBase()->GetAsITKBaseType());

  /** Create an setup Jacobian generator. */
  const auto jacGenerator = JacobianGeneratorType::New();
  jacGenerator->SetTransform(const_cast<const ITKBaseType *>(this->GetAsITKBaseType()));
  jacGenerator->SetOutputSize(resampleImageFilter.GetSize());
  jacGenerator->SetOutputSpacing(resampleImageFilter.GetOutputSpacing());
  jacGenerator->SetOutputOrigin(resampleImageFilter.GetOutputOrigin());
  jacGenerator->SetOutputIndex(resampleImageFilter.GetOutputStartIndex());
  jacGenerator->SetOutputDirection(resampleImageFilter.GetOutputDirection());
  // NOTE: We can not use the following, since the fixed image does not exist in transformix
  //   jacGenerator->SetOutputParametersFromImage(
  //     this->GetRegistration()->GetAsITKBaseType()->GetFixedImage() );

  /** Possibly change direction cosines to their original value, as specified
   * in the tp-file, or by the fixed image. This is only necessary when
   * the UseDirectionCosines flag was set to false.
   */
  const auto              infoChanger = ChangeInfoFilterType::New();
  FixedImageDirectionType originalDirection;
  bool                    retdc = this->GetElastix()->GetOriginalFixedImageDirection(originalDirection);
  infoChanger->SetOutputDirection(originalDirection);
  infoChanger->SetChangeDirection(retdc & !this->GetElastix()->GetUseDirectionCosines());
  infoChanger->SetInput(jacGenerator->GetOutput());

  const auto progressObserver =
    BaseComponent::IsElastixLibrary() ? nullptr : ProgressCommandType::CreateAndConnect(*jacGenerator);
  /** Create a name for the deformation field file. */
  std::string resultImageFormat = "mhd";
  this->m_Configuration->ReadParameter(resultImageFormat, "ResultImageFormat", 0, false);
  std::ostringstream makeFileName;
  makeFileName << this->m_Configuration->GetCommandLineArgument("-out") << "fullSpatialJacobian." << resultImageFormat;

  /** Write outputImage to disk. */
  const auto jacWriter = itk::ImageFileWriter<JacobianImageType>::New();
  jacWriter->SetInput(infoChanger->GetOutput());
  jacWriter->SetFileName(makeFileName.str().c_str());

  // This class is used for writing the fullSpatialJacobian image. It is a hack to ensure that a matrix image is seen as
  // a vector image, which most IO classes understand.
  class PixelTypeChangeCommand : public itk::Command
  {
  public:
    ITK_DISALLOW_COPY_AND_MOVE(PixelTypeChangeCommand);

    /** Standard class typedefs. */
    using Self = PixelTypeChangeCommand;
    using Pointer = itk::SmartPointer<Self>;

    /** Run-time type information (and related methods). */
    itkTypeMacro(PixelTypeChangeCommand, Command);

    /** Method for creation through the object factory. */
    itkNewMacro(Self);

  private:
    using PrivateJacobianImageType = JacobianImageType;

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
  elxout << "  Computing and writing the spatial Jacobian..." << std::endl;
  try
  {
    jacWriter->Update();
  }
  catch (itk::ExceptionObject & excp)
  {
    /** Add information to the exception. */
    excp.SetLocation("TransformBase - ComputeSpatialJacobian()");
    std::string err_str = excp.GetDescription();
    err_str += "\nError occurred while writing spatial Jacobian image.\n";
    excp.SetDescription(err_str);

    /** Pass the exception to an higher level. */
    throw;
  }

} // end ComputeSpatialJacobian()


/**
 * ************** SetTransformParametersFileName ****************
 */

template <class TElastix>
void
TransformBase<TElastix>::SetTransformParametersFileName(const char * filename)
{
  /** Copied from itkSetStringMacro. */
  if (filename && (filename == this->m_TransformParametersFileName))
  {
    return;
  }
  if (filename)
  {
    this->m_TransformParametersFileName = filename;
  }
  else
  {
    this->m_TransformParametersFileName = "";
  }
  this->GetAsITKBaseType()->Modified();

} // end SetTransformParametersFileName()


/**
 * ************** SetReadWriteTransformParameters ***************
 */

template <class TElastix>
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

template <class TElastix>
void
TransformBase<TElastix>::AutomaticScalesEstimation(ScalesType & scales) const
{
  using ImageSamplerType = itk::ImageGridSampler<FixedImageType>;
  using ImageSampleContainerType = typename ImageSamplerType::ImageSampleContainerType;
  using ImageSampleContainerPointer = typename ImageSampleContainerType::Pointer;
  using JacobianType = typename ITKBaseType::JacobianType;
  using NonZeroJacobianIndicesType = typename ITKBaseType::NonZeroJacobianIndicesType;

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
    itkExceptionMacro(<< "No valid voxels found to estimate the scales.");
  }

  /** initialize */
  scales.Fill(0.0);

  /** Read fixed coordinates and get Jacobian. */
  for (const auto & sample : *sampleContainer)
  {
    const InputPointType & point = sample.m_ImageCoordinates;
    // const JacobianType & jacobian = thisITK->GetJacobian( point );
    JacobianType               jacobian;
    NonZeroJacobianIndicesType nzji;
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

template <class TElastix>
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
  using JacobianType = typename ITKBaseType::JacobianType;
  using NonZeroJacobianIndicesType = typename ITKBaseType::NonZeroJacobianIndicesType;

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

  elxout << "start region for scales: " << start << std::endl;
  elxout << "size region for scales: " << size << std::endl;

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
    itkExceptionMacro(<< "No valid voxels found to estimate the scales.");
  }

  /** Read fixed coordinates and get Jacobian. */
  JacobianType               jacobian;
  NonZeroJacobianIndicesType nzji;
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
