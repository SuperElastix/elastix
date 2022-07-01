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
#ifndef elxSplineKernelTransform_hxx
#define elxSplineKernelTransform_hxx

#include "elxSplineKernelTransform.h"
#include "itkTransformixInputPointFileReader.h"
#include <vnl/vnl_math.h>
#include "itkTimeProbe.h"

namespace elastix
{

/**
 * ********************* Constructor ****************************
 */

template <class TElastix>
SplineKernelTransform<TElastix>::SplineKernelTransform()
{
  this->SetKernelType("unknown");
} // end Constructor


/*
 * ******************* SetKernelType ***********************
 */

template <class TElastix>
bool
SplineKernelTransform<TElastix>::SetKernelType(const std::string & kernelType)
{
  this->m_SplineKernelType = kernelType;

  /** According to VTK documentation the R2logR version is
   * appropriate for 2D and the normal for 3D
   * \todo: understand why
   */
  if (SpaceDimension == 2)
  {
    /** only one variant for 2D possible: */
    this->m_KernelTransform = TPRKernelTransformType::New();
  }
  else
  {
    /** 3D: choose between different spline types.
     * \todo: devise one for 4D
     */
    if (kernelType == "ThinPlateSpline")
    {
      this->m_KernelTransform = TPKernelTransformType::New();
    }
    //     else if ( kernelType == "ThinPlateR2LogRSpline" )
    //     {
    //       this->m_KernelTransform = TPRKernelTransformType::New();
    //     }
    else if (kernelType == "VolumeSpline")
    {
      this->m_KernelTransform = VKernelTransformType::New();
    }
    else if (kernelType == "ElasticBodySpline")
    {
      this->m_KernelTransform = EBKernelTransformType::New();
    }
    else if (kernelType == "ElasticBodyReciprocalSpline")
    {
      this->m_KernelTransform = EBRKernelTransformType::New();
    }
    else
    {
      /** unknown kernelType */
      this->m_KernelTransform = KernelTransformType::New();
      return false;
    }
  }

  this->SetCurrentTransform(this->m_KernelTransform);
  return true;

} // end SetKernelType()


/*
 * ******************* BeforeAll ***********************
 */

template <class TElastix>
int
SplineKernelTransform<TElastix>::BeforeAll()
{
  /** Check if -fp is given */
  /** If the optional command "-fp" is given in the command
   * line arguments, then and only then we continue.
   */
  // fp used to be ipp, added in elastix 4.5
  std::string ipp = this->GetConfiguration()->GetCommandLineArgument("-ipp");
  std::string fp = this->GetConfiguration()->GetCommandLineArgument("-fp");

  // Backwards compatibility stuff:
  if (!ipp.empty())
  {
    xl::xout["warning"] << "WARNING: -ipp is deprecated, use -fp instead." << std::endl;
    fp = ipp;
  }

  /** Is the fixed landmark file specified? */
  if (ipp.empty() && fp.empty())
  {
    xl::xout["error"] << "ERROR: -fp should be given for " << this->elxGetClassName()
                      << " in order to define the fixed image (source) landmarks." << std::endl;
    return 1;
  }
  else
  {
    elxout << "-fp       " << fp << std::endl;
  }

  /** Check if -mp is given. If the optional command "-mp"
   * is given in the command line arguments, then we print it.
   */
  std::string mp = this->GetConfiguration()->GetCommandLineArgument("-mp");

  /** Is the moving landmark file specified? */
  if (mp.empty())
  {
    elxout << "-mp       unspecified, assumed equal to -fp" << std::endl;
  }
  else
  {
    elxout << "-mp       " << mp << std::endl;
  }

  return 0;

} // end BeforeAll()


/*
 * ******************* BeforeRegistration ***********************
 */

template <class TElastix>
void
SplineKernelTransform<TElastix>::BeforeRegistration()
{
  /** Determine type of spline. */
  std::string kernelType = "ThinPlateSpline";
  this->GetConfiguration()->ReadParameter(kernelType, "SplineKernelType", this->GetComponentLabel(), 0, -1);
  bool knownType = this->SetKernelType(kernelType);
  if (!knownType)
  {
    xl::xout["error"] << "ERROR: The kernel type " << kernelType << " is not supported." << std::endl;
    itkExceptionMacro(<< "ERROR: unable to configure " << this->GetComponentLabel());
  }

  /** Interpolating or approximating spline. */
  double splineRelaxationFactor = 0.0;
  this->GetConfiguration()->ReadParameter(
    splineRelaxationFactor, "SplineRelaxationFactor", this->GetComponentLabel(), 0, -1);
  this->m_KernelTransform->SetStiffness(splineRelaxationFactor);

  /** Set the Poisson ratio; default = 0.3 = steel. */
  if (kernelType == "ElasticBodySpline" || kernelType == "ElastixBodyReciprocalSpline")
  {
    double poissonRatio = 0.3;
    this->GetConfiguration()->ReadParameter(poissonRatio, "SplinePoissonRatio", this->GetComponentLabel(), 0, -1);
    this->m_KernelTransform->SetPoissonRatio(poissonRatio);
  }

  /** Set the matrix inversion method (one of {SVD, QR}). */
  std::string matrixInversionMethod = "SVD";
  this->GetConfiguration()->ReadParameter(matrixInversionMethod, "TPSMatrixInversionMethod", 0, true);
  this->m_KernelTransform->SetMatrixInversionMethod(matrixInversionMethod);

  /** Load fixed image (source) landmark positions. */
  this->DetermineSourceLandmarks();

  /** Load moving image (target) landmark positions. */
  bool movingLandmarksGiven = this->DetermineTargetLandmarks();

  /** Set all parameters to identity if no moving landmarks were given. */
  if (!movingLandmarksGiven)
  {
    this->m_KernelTransform->SetIdentity();
  }

  /** Set the initial parameters in this->m_Registration. */
  this->m_Registration->GetAsITKBaseType()->SetInitialTransformParameters(this->GetParameters());

  /** \todo: builtin some multiresolution in this transform. */

} // end BeforeRegistration()


/**
 * ************************* DetermineSourceLandmarks *********************
 */

template <class TElastix>
void
SplineKernelTransform<TElastix>::DetermineSourceLandmarks()
{
  /** Load the fixed image landmarks. */
  elxout << "Loading fixed image landmarks for " << this->GetComponentLabel() << ":" << this->elxGetClassName() << "."
         << std::endl;

  // fp used to be ipp
  std::string ipp = this->GetConfiguration()->GetCommandLineArgument("-ipp");
  std::string fp = this->GetConfiguration()->GetCommandLineArgument("-fp");
  if (fp.empty())
  {
    fp = ipp; // backwards compatibility, added in elastix 4.5
  }
  PointSetPointer landmarkPointSet; // default-constructed (null)
  this->ReadLandmarkFile(fp, landmarkPointSet, true);

  /** Set the fp as source landmarks. */
  itk::TimeProbe timer;
  timer.Start();
  elxout << "  Setting the fixed image landmarks (requiring large matrix inversion) ..." << std::endl;
  this->m_KernelTransform->SetSourceLandmarks(landmarkPointSet);
  timer.Stop();
  elxout << "  Setting the fixed image landmarks took: " << Conversion::SecondsToDHMS(timer.GetMean(), 6) << std::endl;

} // end DetermineSourceLandmarks()


/**
 * ************************* DetermineTargetLandmarks *********************
 */

template <class TElastix>
bool
SplineKernelTransform<TElastix>::DetermineTargetLandmarks()
{
  /** The moving landmark file name. */
  std::string mp = this->GetConfiguration()->GetCommandLineArgument("-mp");
  if (mp.empty())
  {
    return false;
  }

  /** Load the moving image landmarks. */
  elxout << "Loading moving image landmarks for " << this->GetComponentLabel() << ":" << this->elxGetClassName() << "."
         << std::endl;

  PointSetPointer landmarkPointSet;
  this->ReadLandmarkFile(mp, landmarkPointSet, false);

  /** Set the mp as target landmarks. */
  itk::TimeProbe timer;
  timer.Start();
  elxout << "  Setting the moving image landmarks ..." << std::endl;
  this->m_KernelTransform->SetTargetLandmarks(landmarkPointSet);
  timer.Stop();
  elxout << "  Setting the moving image landmarks took: " << Conversion::SecondsToDHMS(timer.GetMean(), 6) << std::endl;

  return true;

} // end DetermineTargetLandmarks()


/**
 * ************************* ReadLandmarkFile *********************
 */

template <class TElastix>
void
SplineKernelTransform<TElastix>::ReadLandmarkFile(const std::string & filename,
                                                  PointSetPointer &   landmarkPointSet,
                                                  const bool          landmarksInFixedImage)
{
  /** Typedef's. */
  using IndexType = typename FixedImageType::IndexType;
  using IndexValueType = typename IndexType::IndexValueType;

  /** Construct a landmark file reader and read the points. */
  auto landmarkReader = itk::TransformixInputPointFileReader<PointSetType>::New();
  landmarkReader->SetFileName(filename.c_str());
  try
  {
    landmarkReader->Update();
  }
  catch (const itk::ExceptionObject & err)
  {
    xl::xout["error"] << "  Error while opening landmark file." << std::endl;
    xl::xout["error"] << err << std::endl;
    itkExceptionMacro(<< "ERROR: unable to configure " << this->GetComponentLabel());
  }

  /** Some user-feedback. */
  if (landmarkReader->GetPointsAreIndices())
  {
    elxout << "  Landmarks are specified as image indices." << std::endl;
  }
  else
  {
    elxout << "  Landmarks are specified in world coordinates." << std::endl;
  }
  const unsigned int nrofpoints = landmarkReader->GetNumberOfPoints();
  elxout << "  Number of specified input points: " << nrofpoints << std::endl;

  /** Get the set of input points. */
  landmarkPointSet = landmarkReader->GetOutput();

  /** Convert from index to point if necessary */
  landmarkPointSet->DisconnectPipeline();
  if (landmarkReader->GetPointsAreIndices())
  {
    /** Get handles to the fixed and moving images. */
    typename FixedImageType::Pointer  fixedImage = this->GetElastix()->GetFixedImage();
    typename MovingImageType::Pointer movingImage = this->GetElastix()->GetMovingImage();

    InputPointType landmarkPoint;
    landmarkPoint.Fill(0.0f);
    IndexType landmarkIndex;
    for (unsigned int j = 0; j < nrofpoints; ++j)
    {
      /** The read point from the inputPointSet is actually an index
       * Cast to the proper type.
       */
      landmarkPointSet->GetPoint(j, &landmarkPoint);
      for (unsigned int d = 0; d < SpaceDimension; ++d)
      {
        landmarkIndex[d] = static_cast<IndexValueType>(itk::Math::Round<double>(landmarkPoint[d]));
      }

      /** Compute the input point in physical coordinates and replace the point. */
      if (landmarksInFixedImage)
      {
        fixedImage->TransformIndexToPhysicalPoint(landmarkIndex, landmarkPoint);
      }
      else
      {
        movingImage->TransformIndexToPhysicalPoint(landmarkIndex, landmarkPoint);
      }
      landmarkPointSet->SetPoint(j, landmarkPoint);
    }
  }

  /** Apply initial transform if necessary, for fixed image landmarks only. */
  if (landmarksInFixedImage && this->GetUseComposition() && this->Superclass1::GetInitialTransform() != nullptr)
  {
    InputPointType inputPoint;
    inputPoint.Fill(0.0f);
    for (unsigned int j = 0; j < nrofpoints; ++j)
    {
      landmarkPointSet->GetPoint(j, &inputPoint);
      inputPoint = this->Superclass1::GetInitialTransform()->TransformPoint(inputPoint);
      landmarkPointSet->SetPoint(j, inputPoint);
    }
  }

} // end ReadLandmarkFile()


/**
 * ************************* ReadFromFile ************************
 */

template <class TElastix>
void
SplineKernelTransform<TElastix>::ReadFromFile()
{
  /** Read kernel type. */
  std::string kernelType = "unknown";
  bool        skret = this->GetConfiguration()->ReadParameter(kernelType, "SplineKernelType", 0);
  if (skret)
  {
    this->SetKernelType(kernelType);
  }
  else
  {
    xl::xout["error"] << "ERROR: the SplineKernelType is not given in the transform parameter file." << std::endl;
    itkExceptionMacro(<< "ERROR: unable to configure transform.");
  }

  /** Interpolating or approximating spline. */
  double splineRelaxationFactor = 0.0;
  this->GetConfiguration()->ReadParameter(
    splineRelaxationFactor, "SplineRelaxationFactor", this->GetComponentLabel(), 0, -1);
  this->m_KernelTransform->SetStiffness(splineRelaxationFactor);

  /** Set the Poisson ratio; default = 0.3 = steel. */
  double poissonRatio = 0.3;
  this->GetConfiguration()->ReadParameter(poissonRatio, "SplinePoissonRatio", this->GetComponentLabel(), 0, -1);
  this->m_KernelTransform->SetPoissonRatio(poissonRatio);

  /** Read number of parameters. */
  unsigned int numberOfParameters = 0;
  this->GetConfiguration()->ReadParameter(numberOfParameters, "NumberOfParameters", 0);

  /** Read source landmarks. */
  std::vector<CoordRepType> fixedImageLandmarks(numberOfParameters, itk::NumericTraits<CoordRepType>::ZeroValue());
  bool                      retfil = this->GetConfiguration()->ReadParameter(
    fixedImageLandmarks, "FixedImageLandmarks", 0, numberOfParameters - 1, true);
  if (!retfil)
  {
    xl::xout["error"] << "ERROR: the FixedImageLandmarks are not given in the transform parameter file." << std::endl;
    itkExceptionMacro(<< "ERROR: unable to configure transform.");
  }

  /** Convert to fixedParameters type and set in transform. */
  ParametersType fixedParams(numberOfParameters);
  for (unsigned int i = 0; i < numberOfParameters; ++i)
  {
    fixedParams[i] = fixedImageLandmarks[i];
  }
  this->m_KernelTransform->SetFixedParameters(fixedParams);

  /** Call the ReadFromFile from the TransformBase.
   * This must be done after setting the source landmarks and the
   * splinekerneltype, because later the ReadFromFile from
   * TransformBase calls SetParameters.
   */
  this->Superclass2::ReadFromFile();

} // ReadFromFile()


/**
 * ************************* CustomizeTransformParametersMap ************************
 */

template <class TElastix>
auto
SplineKernelTransform<TElastix>::CreateDerivedTransformParametersMap() const -> ParameterMapType
{
  auto & itkTransform = *m_KernelTransform;

  return { { "SplineKernelType", { m_SplineKernelType } },
           { "SplinePoissonRatio", { Conversion::ToString(itkTransform.GetPoissonRatio()) } },
           { "SplineRelaxationFactor", { Conversion::ToString(itkTransform.GetStiffness()) } },
           { "FixedImageLandmarks", Conversion::ToVectorOfStrings(itkTransform.GetFixedParameters()) } };

} // end CustomizeTransformParametersMap()


} // end namespace elastix

#endif // end #ifndef elxSplineKernelTransform_hxx
