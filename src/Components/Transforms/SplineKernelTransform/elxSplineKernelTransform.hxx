/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __elxSplineKernelTransform_HXX_
#define __elxSplineKernelTransform_HXX_

#include "elxSplineKernelTransform.h"
#include "itkTransformixInputPointFileReader.h"
#include "vnl/vnl_math.h"

namespace elastix
{
using namespace itk;

/**
 * ********************* Constructor ****************************
 */

template <class TElastix>
SplineKernelTransform<TElastix>
::SplineKernelTransform()
{
  this->SetKernelType( "unknown" );
} // end Constructor


/*
 * ******************* SetKernelType ***********************
 */

template <class TElastix>
bool
SplineKernelTransform<TElastix>
::SetKernelType( const std::string & kernelType )
{
  this->m_SplineKernelType = kernelType;

  /** According to VTK documentation the R2logR version is
   * appropriate for 2D and the normal for 3D
   * \todo: understand why
   */
  if ( SpaceDimension == 2 )
  {
    /** only one variant for 2D possible: */
    this->m_KernelTransform = TPRKernelTransformType::New();
  }
  else
  {
    /** 3D: choose between different spline types.
     * \todo: devise one for 4D
     */
    if ( kernelType == "ThinPlateSpline" )
    {
      this->m_KernelTransform = TPKernelTransformType::New();
    }
//     else if ( kernelType == "ThinPlateR2LogRSpline" )
//     {
//       this->m_KernelTransform = TPRKernelTransformType::New();
//     }
    else if ( kernelType == "VolumeSpline" )
    {
      this->m_KernelTransform = VKernelTransformType::New();
    }
    else if ( kernelType == "ElasticBodySpline" )
    {
      this->m_KernelTransform = EBKernelTransformType::New();
    }
    else if ( kernelType == "ElasticBodyReciprocalSpline" )
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

  this->SetCurrentTransform( this->m_KernelTransform );
  return true;

} // end SetKernelType()


/*
 * ******************* BeforeAll ***********************
 */

template <class TElastix>
int
SplineKernelTransform<TElastix>
::BeforeAll( void )
{
  /** Check if -ipp is given */
  /** If the optional command "-ipp" is given in the command
   * line arguments, then and only then we continue.
   */
  std::string ipp = this->GetConfiguration()->GetCommandLineArgument( "-ipp" );

  /** If there is an inputpoint-file? */
  if ( ipp.empty() )
  {
    xl::xout["error"]
      << "ERROR: -ipp should be given for "
      << this->elxGetClassName()
      << " in order to define the source landmarks."
      << std::endl;
      return 1;
  }
  else
  {
    elxout << "-ipp      " << ipp << std::endl;
    return 0;
  }

} // end BeforeAll()


/*
 * ******************* BeforeRegistration ***********************
 */

template <class TElastix>
void
SplineKernelTransform<TElastix>
::BeforeRegistration( void )
{
  /** Determine type of spline. */
  std::string kernelType = "ThinPlateSpline";
  this->GetConfiguration()->ReadParameter(
    kernelType, "SplineKernelType", this->GetComponentLabel(), 0, -1 );
  bool knownType = this->SetKernelType( kernelType );
  if ( !knownType )
  {
    xl::xout["error"] << "ERROR: The kernel type " << kernelType
      << " is not supported." << std::endl;
    itkExceptionMacro( << "ERROR: unable to configure "
      << this->GetComponentLabel() );
  }

  /** Interpolating or approximating spline. */
  double splineRelaxationFactor = 0.0;
  this->GetConfiguration()->ReadParameter(
    splineRelaxationFactor, "SplineRelaxationFactor", this->GetComponentLabel(), 0, -1 );
  this->m_KernelTransform->SetStiffness( splineRelaxationFactor );

  /** Set the Poisson ratio; default = 0.3 = steel. */
  if ( kernelType == "ElasticBodySpline" || kernelType == "ElastixBodyReciprocalSpline" )
  {
    double poissonRatio = 0.3;
    this->GetConfiguration()->ReadParameter(
      poissonRatio, "SplinePoissonRatio", this->GetComponentLabel(), 0, -1 );
    this->m_KernelTransform->SetPoissonRatio( poissonRatio );
  }

  /** Load source landmark positions. */
  this->DetermineSourceLandmarks();

  /** Set all parameters to zero. */
  this->m_KernelTransform->SetIdentity();

  /** Set the initial parameters in this->m_Registration. */
  this->m_Registration->GetAsITKBaseType()
    ->SetInitialTransformParameters( this->GetParameters() );

  /** \todo: builtin some multiresolution in this transform. */

} // end BeforeRegistration()


/**
 * ************************* DetermineSourceLandmarks *********************
 */

template <class TElastix>
void
SplineKernelTransform<TElastix>
::DetermineSourceLandmarks( void )
{
  /** Typedef's. */
  typedef typename FixedImageType::IndexType            FixedImageIndexType;
  typedef typename FixedImageIndexType::IndexValueType  FixedImageIndexValueType;
  typedef typename KernelTransformType::PointSetType            PointSetType;
  typedef itk::TransformixInputPointFileReader<
    PointSetType >                                      IPPReaderType;

  elxout << "Loading fixed image landmarks for " << this->GetComponentLabel()
    << ":" << this->elxGetClassName() << "." << std::endl;

  /** Construct an ipp-file reader and read the points. */
  typename IPPReaderType::Pointer ippReader = IPPReaderType::New();
  std::string ipp = this->GetConfiguration()->GetCommandLineArgument( "-ipp" );
  ippReader->SetFileName( ipp.c_str() );
  elxout << "  Reading input point file: " << ipp << std::endl;
  try
  {
    ippReader->Update();
  }
  catch ( itk::ExceptionObject & err )
  {
    xl::xout["error"] << "  Error while opening input point file." << std::endl;
    xl::xout["error"] << err << std::endl;
    itkExceptionMacro( << "ERROR: unable to configure " << this->GetComponentLabel() );
  }

  /** Some user-feedback. */
  if ( ippReader->GetPointsAreIndices() )
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

  /** Convert from index to point if necessary */
  inputPointSet->DisconnectPipeline();
  if ( ippReader->GetPointsAreIndices() )
  {
    /** Convert to world coordinates. */
    typename FixedImageType::Pointer fixedImage = this->GetElastix()->GetFixedImage();
    InputPointType inputPoint; inputPoint.Fill( 0.0f );
    FixedImageIndexType inputIndex;
    for ( unsigned int j = 0; j < nrofpoints; ++j )
    {
      /** The read point from the inputPointSet is actually an index
       * Cast to the proper type.
       */
      inputPointSet->GetPoint( j, &inputPoint );
      for ( unsigned int d = 0; d < SpaceDimension; ++d )
      {
        inputIndex[ d ] = static_cast<FixedImageIndexValueType>(
          vnl_math_rnd( inputPoint[ d ] ) );
      }

      /** Compute the input point in physical coordinates. */
      fixedImage->TransformIndexToPhysicalPoint(
        inputIndex, inputPoint );
      inputPointSet->SetPoint( j, inputPoint );
    }
  }

  /** Apply initial transform if necessary. */
  if ( this->GetUseComposition()
    && this->Superclass1::GetInitialTransform() != 0 )
  {
    InputPointType inputPoint; inputPoint.Fill( 0.0f );
    for ( unsigned int j = 0; j < nrofpoints; ++j )
    {
      inputPointSet->GetPoint( j, &inputPoint );
      inputPoint = this->Superclass1::GetInitialTransform()
        ->TransformPoint( inputPoint );
      inputPointSet->SetPoint( j, inputPoint );
    }
  }

  /** Set the ipp as source landmarks. */
  this->m_KernelTransform->SetSourceLandmarks( inputPointSet );

} // end DetermineSourceLandmarks()


/**
 * ************************* ReadFromFile ************************
 */

template <class TElastix>
void
SplineKernelTransform<TElastix>
::ReadFromFile( void )
{
  /** Read kernel type. */
  std::string kernelType = "unknown";
  bool skret = this->GetConfiguration()->ReadParameter(
    kernelType, "SplineKernelType", 0 );
  if ( skret )
  {
    this->SetKernelType( kernelType );
  }
  else
  {
    xl::xout["error"] << "ERROR: the SplineKernelType is not given in the "
      << "transform parameter file." << std::endl;
    itkExceptionMacro( << "ERROR: unable to configure transform." );
  }

  /** Interpolating or approximating spline. */
  double splineRelaxationFactor = 0.0;
  this->GetConfiguration()->ReadParameter(
    splineRelaxationFactor, "SplineRelaxationFactor", this->GetComponentLabel(), 0, -1 );
  this->m_KernelTransform->SetStiffness( splineRelaxationFactor );

  /** Set the Poisson ratio; default = 0.3 = steel. */
  double poissonRatio = 0.3;
  this->GetConfiguration()->ReadParameter(
    poissonRatio, "SplinePoissonRatio", this->GetComponentLabel(), 0, -1 );
  this->m_KernelTransform->SetPoissonRatio( poissonRatio );

  /** Read number of parameters. */
  unsigned int numberOfParameters = 0;
  this->GetConfiguration()->ReadParameter(
    numberOfParameters, "NumberOfParameters", 0 );

  /** Read source landmarks. */
  std::vector<CoordRepType> fixedImageLandmarks(
    numberOfParameters, itk::NumericTraits<CoordRepType>::Zero );
  bool retfil = this->GetConfiguration()->ReadParameter(
    fixedImageLandmarks, "FixedImageLandmarks", 0, numberOfParameters - 1, true );
  if ( ! retfil )
  {
    xl::xout["error"] << "ERROR: the FixedImageLandmarks are not given in "
      << "the transform parameter file." << std::endl;
    itkExceptionMacro( << "ERROR: unable to configure transform." );
  }

  /** Convert to fixedParameters type and set in transform. */
  ParametersType fixedParams( numberOfParameters );
  for ( unsigned int i = 0; i < numberOfParameters; ++i )
  {
    fixedParams[ i ] = fixedImageLandmarks[ i ];
  }
  this->m_KernelTransform->SetFixedParameters( fixedParams );

  /** Call the ReadFromFile from the TransformBase.
   * This must be done after setting the source landmarks and the
   * splinekerneltype, because later the ReadFromFile from
   * TransformBase calls SetParameters.
   */
  this->Superclass2::ReadFromFile();

} // ReadFromFile()


/**
 * ************************* WriteToFile ************************
 * Save the kernel type and the source landmarks
 */

template <class TElastix>
void
SplineKernelTransform<TElastix>
::WriteToFile( const ParametersType & param ) const
{
  /** Call the WriteToFile from the TransformBase. */
  this->Superclass2::WriteToFile( param );

  /** Add some SplineKernelTransform specific lines. */
  xl::xout["transpar"] << std::endl << "// SplineKernelTransform specific" << std::endl;

  /** Write the SplineKernelType of this transform. */
  xl::xout["transpar"] << "(SplineKernelType \""
    << this->m_SplineKernelType << "\")" << std::endl;

  /** Write the relaxation and Poisson ratio parameters. */
  xl::xout["transpar"] << "(SplinePoissonRatio "
    << this->m_KernelTransform->GetPoissonRatio() << ")" << std::endl;
  xl::xout["transpar"] << "(SplineRelaxationFactor "
    << this->m_KernelTransform->GetStiffness() << ")" << std::endl;

  /** Write the fixed image landmarks. */
  const ParametersType & fixedParams = this->m_KernelTransform->GetFixedParameters();
  xl::xout["transpar"] << "(FixedImageLandmarks ";
  for ( unsigned int i = 0; i < fixedParams.GetSize() - 1; ++i )
  {
    xl::xout["transpar"] << fixedParams[i] << " ";
  }
  xl::xout["transpar"] << fixedParams[ fixedParams.GetSize() - 1 ] << ")" << std::endl;

} // end WriteToFile()


} // end namespace elastix


#endif // end #ifndef __elxSplineKernelTransform_HXX_
