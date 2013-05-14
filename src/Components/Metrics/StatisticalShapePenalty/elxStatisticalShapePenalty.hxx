/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

  If you use the StatisticalShapePenalty anywhere we would appreciate if you cite the following article:
  F.F. Berendsen et al., Free-form image registration regularized by a statistical shape model: application to organ segmentation in cervical MR, Comput. Vis. Image Understand. (2013), http://dx.doi.org/10.1016/j.cviu.2012.12.006

======================================================================*/

#ifndef __elxStatisticalShapePenalty_HXX__
#define __elxStatisticalShapePenalty_HXX__

#include "elxStatisticalShapePenalty.h"
#include "itkTransformixInputPointFileReader.h"

#include "itkPointSet.h"
#include "itkDefaultStaticMeshTraits.h"
#include "itkVTKPolyDataReader.h"
#include "itkVTKPolyDataWriter.h"
#include "itkTransformMeshFilter.h"
#include <itkMesh.h>

#include <typeinfo>

namespace elastix
{
using namespace itk;

/**
 * ******************* Initialize ***********************
 */

template <class TElastix>
void
StatisticalShapePenalty<TElastix>
::Initialize( void ) throw (ExceptionObject)
{
  TimerPointer timer = TimerType::New();
  timer->StartTimer();
  this->Superclass1::Initialize();
  timer->StopTimer();
  elxout << "Initialization of StatisticalShape metric took: "
    << static_cast<long>( timer->GetElapsedClockSec() * 1000 )
    << " ms." << std::endl;

} // end Initialize()


/**
 * ***************** BeforeRegistration ***********************
 */

template <class TElastix>
void
StatisticalShapePenalty<TElastix>
::BeforeRegistration( void )
{
  //elxout << "Stat PH BeforeRegistration " << std::endl;
   /** Get and set NormalizedShapeModel. Default TRUE. */
  bool normalizedShapeModel = true;
  this->GetConfiguration()->ReadParameter( normalizedShapeModel, "NormalizedShapeModel", 0,0);
  this->SetNormalizedShapeModel( normalizedShapeModel );
  //elxout << "NormalizedShapeModel: " << this->GetNormalizedShapeModel() << std::endl;

   /** Get and set NormalizedShapeModel. Default TRUE. */
  int shapeModelCalculation = 0;
  this->GetConfiguration()->ReadParameter( shapeModelCalculation, "ShapeModelCalculation", 0,0);
  this->SetShapeModelCalculation( shapeModelCalculation );
  //elxout << "ShapeModelCalculation: " << this->GetShapeModelCalculation() << std::endl;


  /** Read and set the fixed pointset. */
  std::string fixedName = this->GetConfiguration()->GetCommandLineArgument( "-fp" );
  typename PointSetType::Pointer fixedPointSet = 0;
  const typename ImageType::ConstPointer fixedImage = this->GetElastix()->GetFixedImage();
  const unsigned int nrOfFixedPoints = this->ReadShape(
    fixedName, fixedPointSet, fixedImage );
  this->SetFixedPointSet( fixedPointSet );

  // itkCombinationImageToImageMetric.txx checks if metric base class is ImageMetricType or PointSetMetricType.
  // This class is derived from SingleValuedPointSetToPointSetMetric which needs a moving pointset.
  this->SetMovingPointSet( fixedPointSet ); // TODO: make itkCombinationImageToImageMetric check for a base class metric that doesn't use an image or moving pointset.
 
  /** Read meanVector filename. */
  std::string meanVectorName = this->GetConfiguration()->GetCommandLineArgument( "-mean" ); 
  vcl_ifstream datafile;
  vnl_vector<double>* const meanVector = new vnl_vector<double>();
  datafile.open(meanVectorName.c_str());
  if (datafile.is_open())
  {
      meanVector->read_ascii(datafile);
      datafile.close();
      datafile.clear();
      elxout << " meanVector " << meanVectorName << " read" << std::endl;
  }
  else
  {
    itkExceptionMacro( << "Unable to open meanVector file: " << meanVectorName);
  }
  this->SetMeanVector(meanVector);
  
  /** Check. */
  if ( normalizedShapeModel )
  {
    if ( nrOfFixedPoints*Self::FixedPointSetDimension != meanVector->size()-Self::FixedPointSetDimension-1 )
    {
      itkExceptionMacro( << "ERROR: the number of elements in the meanVector (" << meanVector->size()
        << ") does not match the number of points of the fixed pointset ("
        << nrOfFixedPoints << ") times the point dimensionality (" <<
        Self::FixedPointSetDimension << ") plus a Centroid of dimension " <<
        Self::FixedPointSetDimension << " plus a size element");
    }
  } 
  else
  {
    if ( nrOfFixedPoints*Self::FixedPointSetDimension != meanVector->size())
    {
      itkExceptionMacro( << "ERROR: the number of elements in the meanVector (" << meanVector->size()
        << ") does not match the number of points of the fixed pointset ("
        << nrOfFixedPoints << ") times the point dimensionality (" <<
        Self::FixedPointSetDimension << ")");
    }
  }

  /** Read covariancematrix filename. */
  std::string covarianceMatrixName = this->GetConfiguration()->GetCommandLineArgument( "-covariance" ); 

  vnl_matrix<double>* const covarianceMatrix = new vnl_matrix<double>();
  
  datafile.open(covarianceMatrixName.c_str());
  if (datafile.is_open())
  {
      covarianceMatrix->read_ascii(datafile);
      datafile.close();
      datafile.clear();
      elxout << "covarianceMatrix "<< covarianceMatrixName << " read" << std::endl;
  }
  else
  {
    itkExceptionMacro( << "Unable to open covarianceMatrix file: " << covarianceMatrixName);
  }
  this->SetCovarianceMatrix(covarianceMatrix);

  /** Read eigenvectormatrix filename. */
  std::string eigenVectorsName = this->GetConfiguration()->GetCommandLineArgument( "-evectors" ); 

  vnl_matrix<double>* const eigenVectors = new vnl_matrix<double>();
  
  datafile.open(eigenVectorsName.c_str());
  if (datafile.is_open())
  {
      eigenVectors->read_ascii(datafile);
      datafile.close();
      datafile.clear();
      elxout << "eigenvectormatrix "<< eigenVectorsName << " read" << std::endl;
  }
  else
  {
    //itkExceptionMacro( << "Unable to open EigenVectors file: " << eigenVectorsName);
  }
  this->SetEigenVectors(eigenVectors);

  /** Read eigenvaluevector filename. */
  std::string eigenValuesName = this->GetConfiguration()->GetCommandLineArgument( "-evalues" ); 
  vnl_vector<double>* const eigenValues = new vnl_vector<double>();
  datafile.open(eigenValuesName.c_str());
  if (datafile.is_open())
  {
      eigenValues->read_ascii(datafile);
      datafile.close();
      datafile.clear();
      elxout << "eigenvaluevector "<< eigenValuesName << " read" << std::endl;
  }
  else
  {
    //itkExceptionMacro( << "Unable to open EigenValues file: " << eigenValuesName);
  }
  this->SetEigenValues(eigenValues);
  //elxout << "eigenvaluevector read" << std::endl;
  //elxout << "size: " << m_eigenvalues.size() << std::endl;


} // end BeforeRegistration()

/**
 * ***************** BeforeEachResolution ***********************
 */

template <class TElastix>
void
StatisticalShapePenalty<TElastix>
::BeforeEachResolution( void )
{
    /** Get the current resolution level. */
  unsigned int level
    = this->m_Registration->GetAsITKBaseType()->GetCurrentLevel();

  /** Get and set ShrinkageIntensity. Default 0.5. */
  double shrinkageIntensity = 0.5;
  this->GetConfiguration()->ReadParameter( shrinkageIntensity, "ShrinkageIntensity",
  this->GetComponentLabel(), level, 0 );

  if(this->GetShrinkageIntensity()!=shrinkageIntensity)
  {
    this->SetShrinkageIntensityNeedsUpdate( true );
  }
  this->SetShrinkageIntensity( shrinkageIntensity );

  /** Get and set BaseVariance. Default 1000. */
  double baseVariance = 1000;
  this->GetConfiguration()->ReadParameter( baseVariance,
    "BaseVariance", this->GetComponentLabel(), level, 0 );

  if(this->GetBaseVariance()!=baseVariance)
  {
    this->SetBaseVarianceNeedsUpdate( true );
  }
  this->SetBaseVariance( baseVariance );

  /** Get and set CentroidXVariance. Default 10. */
  double centroidXVariance = 10;
  this->GetConfiguration()->ReadParameter( centroidXVariance,
    "CentroidXVariance", this->GetComponentLabel(), level, 0 );

  if(this->GetCentroidXVariance()!=centroidXVariance)
  {
    this->SetVariancesNeedsUpdate( true );
  }
  this->SetCentroidXVariance( centroidXVariance );

  /** Get and set CentroidYVariance. Default 10. */
  double centroidYVariance = 10;
  this->GetConfiguration()->ReadParameter( centroidYVariance,
    "CentroidYVariance", this->GetComponentLabel(), level, 0 );

  if(this->GetCentroidYVariance()!=centroidYVariance)
  {
    this->SetVariancesNeedsUpdate( true );
  }
  this->SetCentroidYVariance( centroidYVariance );

  /** Get and set CentroidZVariance. Default 10. */
  double centroidZVariance = 10;
  this->GetConfiguration()->ReadParameter( centroidZVariance,
    "CentroidZVariance", this->GetComponentLabel(), level, 0 );

  if(this->GetCentroidZVariance()!=centroidZVariance)
  {
    this->SetVariancesNeedsUpdate( true );
  }
  this->SetCentroidZVariance( centroidZVariance );

  /** Get and set SizeVariance. Default 10. */
  double sizeVariance = 10;
  this->GetConfiguration()->ReadParameter( sizeVariance,
    "SizeVariance", this->GetComponentLabel(), level, 0 );

  if(this->GetSizeVariance()!=sizeVariance)
  {
    this->SetVariancesNeedsUpdate( true );
  }
  this->SetSizeVariance( sizeVariance );

  /** Get and set CutOffValue. Default 0. */
  double cutOffValue = 0;
  this->GetConfiguration()->ReadParameter( cutOffValue,
    "CutOffValue", this->GetComponentLabel(), level, 0 );
  this->SetCutOffValue( cutOffValue );
  /** Get and set CutOffSharpness. Default 2. */
  double cutOffSharpness = 2.0;
  this->GetConfiguration()->ReadParameter( cutOffSharpness,
    "CutOffSharpness", this->GetComponentLabel(), level, 0 );
  this->SetCutOffSharpness( cutOffSharpness );
} // end BeforeEachResolution()


/**
 * ***************** ReadLandmarks ***********************
 */

template <class TElastix>
unsigned int
StatisticalShapePenalty<TElastix>
::ReadLandmarks(
  const std::string & landmarkFileName,
  typename PointSetType::Pointer & pointSet,
  const typename ImageType::ConstPointer image )
{
  /** Typedefs. */
  typedef typename ImageType::IndexType       IndexType;
  typedef typename ImageType::IndexValueType  IndexValueType;
  typedef typename ImageType::PointType       PointType;
  typedef itk::TransformixInputPointFileReader<
    PointSetType >                            PointSetReaderType;

  elxout << "Loading landmarks for " << this->GetComponentLabel()
    << ":" << this->elxGetClassName() << "." << std::endl;

  /** Read the landmarks. */
  typename PointSetReaderType::Pointer reader = PointSetReaderType::New();
  reader->SetFileName( landmarkFileName.c_str() );
  elxout << "  Reading landmark file: " << landmarkFileName << std::endl;
  try
  {
    reader->Update();
  }
  catch ( itk::ExceptionObject & err )
  {
    xl::xout["error"] << "  Error while opening " << landmarkFileName << std::endl;
    xl::xout["error"] << err << std::endl;
    itkExceptionMacro( << "ERROR: unable to configure " << this->GetComponentLabel() );
  }

  /** Some user-feedback. */
  const unsigned int nrofpoints = reader->GetNumberOfPoints();
  if ( reader->GetPointsAreIndices() )
  {
    elxout << "  Landmarks are specified as image indices." << std::endl;
  }
  else
  {
    elxout << "  Landmarks are specified in world coordinates." << std::endl;
  }
  elxout << "  Number of specified points: " << nrofpoints << std::endl;

  /** Get the pointset. */
  pointSet = reader->GetOutput();

  /** Convert from index to point if necessary */
  pointSet->DisconnectPipeline();
  if ( reader->GetPointsAreIndices() )
  {
    /** Convert to world coordinates */
    for ( unsigned int j = 0; j < nrofpoints; ++j )
    {
      /** The landmarks from the pointSet are indices. We first cast to the
       * proper type, and then convert it to world coordinates.
       */
      PointType point; IndexType index;
      pointSet->GetPoint( j, &point );
      for ( unsigned int d = 0; d < FixedImageDimension; ++d )
      {
        index[ d ] = static_cast<IndexValueType>( vnl_math_rnd( point[ d ] ) );
      }

      /** Compute the input point in physical coordinates. */
      image->TransformIndexToPhysicalPoint( index, point );
      pointSet->SetPoint( j, point );

    } // end for all points
  } // end for points are indices

  return nrofpoints;

} // end ReadLandmarks()

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
unsigned int
StatisticalShapePenalty<TElastix>
::ReadShape(
  const std::string & ShapeFileName,
  typename PointSetType::Pointer & pointSet,
  const typename ImageType::ConstPointer image )
{
  /** Typedef's. \todo test DummyIPPPixelType=bool. */
  typedef double                                         DummyIPPPixelType;
  //typedef float                                         DummyIPPPixelType;
  //typedef FixedPointSetPixelType                          DummyIPPPixelType;

  typedef DefaultStaticMeshTraits<
    DummyIPPPixelType, FixedImageDimension,
    FixedImageDimension, CoordRepType>                  MeshTraitsType;

  /*typedef DefaultStaticMeshTraits<
    PointSet::PixelType, FixedImageDimension,
    FixedImageDimension, CoordRepType>                  MeshTraitsType;
  */
  typedef Mesh<
    DummyIPPPixelType, FixedImageDimension, MeshTraitsType > MeshType;

  /*typedef Mesh<
    PointSet::PixelType, FixedImageDimension, MeshTraitsType > MeshType;
    */
  typedef VTKPolyDataReader< MeshType >            MeshReaderType;
  //typedef VTKPolyDataReader<PointSetType >            MeshReaderType;

  //typedef VTKPolyDataWriter< MeshType >            MeshWriterType;
  //typename PointSetType::Pointer
//  typedef TransformMeshFilter<
//    MeshType, MeshType, CombinationTransformType>       TransformMeshFilterType;

  /** Read the input points. */
  typename MeshReaderType::Pointer meshReader = MeshReaderType::New();
  meshReader->SetFileName( ShapeFileName.c_str() );
  elxout << "  Reading input point file: " << ShapeFileName << std::endl;
  try
  {
    meshReader->Update();
  }
  catch (ExceptionObject & err)
  {
    xl::xout["error"] << "  Error while opening input point file." << std::endl;
    xl::xout["error"] << err << std::endl;
  }

  /** Some user-feedback. */
  elxout << "  Input points are specified in world coordinates." << std::endl;
  unsigned long nrofpoints = meshReader->GetOutput()->GetNumberOfPoints();
  elxout << "  Number of specified input points: " << nrofpoints << std::endl;


  //elxout << " typeof pointSet: " << typeid(pointSet).name() << std::endl;
  typename MeshType::Pointer mesh = meshReader->GetOutput();
  //elxout << " typeof mesh: " << typeid(mesh).name() << std::endl;

  pointSet = PointSetType::New();
  pointSet->SetPoints(mesh->GetPoints());


  return nrofpoints;
} // end ReadShape()

} // end namespace elastix


#endif // end #ifndef __elxStatisticalShapePenalty_HXX__

