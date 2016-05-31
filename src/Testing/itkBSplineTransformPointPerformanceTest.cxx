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
#include "itkAdvancedBSplineDeformableTransform.h"

#include "itkImageRegionIterator.h"

// Report timings
#include "itkTimeProbe.h"

#include <fstream>
#include <iomanip>

//-------------------------------------------------------------------------------------
// Create a class that inherits from the B-spline transform,
// and adds the previous un-optimized TransformPoint function.
namespace itk
{
template< class TScalarType = double, unsigned int NDimensions = 3, unsigned int VSplineOrder = 3 >
class BSplineTransform_TEST :
  public AdvancedBSplineDeformableTransform< TScalarType, NDimensions, VSplineOrder >
{
public:

  /** Standard class typedefs. */
  typedef BSplineTransform_TEST Self;
  typedef AdvancedBSplineDeformableTransform<
    TScalarType, NDimensions, VSplineOrder >        Superclass;
  typedef SmartPointer< Self >       Pointer;
  typedef SmartPointer< const Self > ConstPointer;

  /** Some stuff that is needed to get this class functional. */
  itkNewMacro( Self );
  itkTypeMacro( BSplineTransform_TEST, AdvancedBSplineDeformableTransform );
  itkStaticConstMacro( SpaceDimension, unsigned int, NDimensions );
  typedef typename Superclass::InputPointType          InputPointType;
  typedef typename Superclass::OutputPointType         OutputPointType;
  typedef typename Superclass::IndexType               IndexType;
  typedef typename Superclass::ContinuousIndexType     ContinuousIndexType;
  typedef typename Superclass::WeightsFunctionType     WeightsFunctionType;
  typedef typename Superclass::WeightsType             WeightsType;
  typedef typename Superclass::ParameterIndexArrayType ParameterIndexArrayType;
  typedef typename Superclass::ImageType               ImageType;
  typedef typename Superclass::RegionType              RegionType;
  typedef typename Superclass::PixelType               PixelType;
  typedef typename Superclass::ScalarType              ScalarType;

  /** Transform points by a B-spline deformable transformation. */
  OutputPointType TransformPoint_OLD( const InputPointType & point ) const
  {
    const unsigned long numberOfWeights = WeightsFunctionType::NumberOfWeights;
    typename WeightsType::ValueType weightsArray[ numberOfWeights ];
    typename ParameterIndexArrayType::ValueType indicesArray[ numberOfWeights ];
    WeightsType             weights( weightsArray, numberOfWeights, false );
    ParameterIndexArrayType indices( indicesArray, numberOfWeights, false );

    OutputPointType outputPoint;  bool inside;
    this->TransformPoint_OLD( point, outputPoint, weights, indices, inside );
    return outputPoint;
  } // end TransformPoint_OLD()


  void TransformPoint_OLD(
    const InputPointType & inputPoint,
    OutputPointType & outputPoint,
    WeightsType & weights,
    ParameterIndexArrayType & indices,
    bool & inside ) const
  {
    inside = true;
    InputPointType transformedPoint = inputPoint;

    /** Check if the coefficient image has been set. */
    if( !this->m_CoefficientImages[ 0 ] )
    {
      itkWarningMacro( << "B-spline coefficients have not been set" );
      for( unsigned int j = 0; j < SpaceDimension; j++ )
      {
        outputPoint[ j ] = transformedPoint[ j ];
      }
      return;
    }

    /***/
    ContinuousIndexType cindex;
    this->TransformPointToContinuousGridIndex( inputPoint, cindex );

    // NOTE: if the support region does not lie totally within the grid
    // we assume zero displacement and return the input point
    inside = this->InsideValidRegion( cindex );
    if( !inside )
    {
      outputPoint = transformedPoint;
      return;
    }

    // Compute interpolation weights
    IndexType supportIndex;
    this->m_WeightsFunction->ComputeStartIndex( cindex, supportIndex );
    this->m_WeightsFunction->Evaluate( cindex, supportIndex, weights );

    // For each dimension, correlate coefficient with weights
    RegionType supportRegion;
    supportRegion.SetSize( this->m_SupportSize );
    supportRegion.SetIndex( supportIndex );

    outputPoint.Fill( NumericTraits< ScalarType >::ZeroValue() );

    /** Create iterators over the coefficient images. */
    typedef ImageRegionConstIterator< ImageType > IteratorType;
    IteratorType      iterator[ SpaceDimension ];
    unsigned long     counter = 0;
    const PixelType * basePointer
      = this->m_CoefficientImages[ 0 ]->GetBufferPointer();

    for( unsigned int j = 0; j < SpaceDimension; j++ )
    {
      iterator[ j ] = IteratorType( this->m_CoefficientImages[ j ], supportRegion );
    }

    /** Loop over the support region. */
    while( !iterator[ 0 ].IsAtEnd() )
    {
      // populate the indices array
      indices[ counter ] = &( iterator[ 0 ].Value() ) - basePointer;

      // multiply weigth with coefficient to compute displacement
      for( unsigned int j = 0; j < SpaceDimension; j++ )
      {
        outputPoint[ j ] += static_cast< ScalarType >(
          weights[ counter ] * iterator[ j ].Value() );
        ++iterator[ j ];
      }
      ++counter;

    } // end while

    // The output point is the start point + displacement.
    for( unsigned int j = 0; j < SpaceDimension; j++ )
    {
      outputPoint[ j ] += transformedPoint[ j ];
    }

  } // end TransformPoint_OLD()


};

// end class BSplineTransform_TEST
} // end namespace itk

//-------------------------------------------------------------------------------------

int
main( int argc, char * argv[] )
{
  /** Some basic type definitions.
   * NOTE: don't change the dimension or the spline order, since the
   * hard-coded ground truth depends on this.
   */
  const unsigned int Dimension   = 3;
  const unsigned int SplineOrder = 3;
  typedef double CoordinateRepresentationType;

  /** The number of calls to Evaluate(). Distinguish between
   * Debug and Release mode.
   */
#ifndef NDEBUG
  unsigned int N = static_cast< unsigned int >( 1e3 );
#else
  unsigned int N = static_cast< unsigned int >( 1e5 );
#endif
  std::cerr << "N = " << N << std::endl;

  /** Check. */
  if( argc != 2 )
  {
    std::cerr << "ERROR: You should specify a text file with the B-spline "
              << "transformation parameters." << std::endl;
    return 1;
  }

  /** Typedefs. */
  typedef itk::BSplineTransform_TEST<
    CoordinateRepresentationType, Dimension, SplineOrder >    TransformType;

  typedef TransformType::InputPointType  InputPointType;
  typedef TransformType::OutputPointType OutputPointType;
  typedef TransformType::ParametersType  ParametersType;

  typedef itk::Image< CoordinateRepresentationType,
    Dimension >                                         InputImageType;
  typedef InputImageType::RegionType    RegionType;
  typedef InputImageType::SizeType      SizeType;
  typedef InputImageType::IndexType     IndexType;
  typedef InputImageType::SpacingType   SpacingType;
  typedef InputImageType::PointType     OriginType;
  typedef InputImageType::DirectionType DirectionType;

  /** Create the transform. */
  TransformType::Pointer transform = TransformType::New();

  /** Setup the B-spline transform:
   * (GridSize 44 43 35)
   * (GridIndex 0 0 0)
   * (GridSpacing 10.7832773148 11.2116431394 11.8648235177)
   * (GridOrigin -237.6759555555 -239.9488431747 -344.2315805162)
   */
  SizeType gridSize;
  gridSize[ 0 ] = 44; gridSize[ 1 ] = 43; gridSize[ 2 ] = 35;
  IndexType gridIndex;
  gridIndex.Fill( 0 );
  RegionType gridRegion;
  gridRegion.SetSize( gridSize );
  gridRegion.SetIndex( gridIndex );
  SpacingType gridSpacing;
  gridSpacing[ 0 ] = 10.7832773148;
  gridSpacing[ 1 ] = 11.2116431394;
  gridSpacing[ 2 ] = 11.8648235177;
  OriginType gridOrigin;
  gridOrigin[ 0 ] = -237.6759555555;
  gridOrigin[ 1 ] = -239.9488431747;
  gridOrigin[ 2 ] = -344.2315805162;
  DirectionType gridDirection;
  gridDirection.SetIdentity();

  transform->SetGridOrigin( gridOrigin );
  transform->SetGridSpacing( gridSpacing );
  transform->SetGridRegion( gridRegion );
  transform->SetGridDirection( gridDirection );

  /** Now read the parameters as defined in the file par.txt. */
  ParametersType parameters( transform->GetNumberOfParameters() );
  std::ifstream  input( argv[ 1 ] );
  if( input.is_open() )
  {
    for( unsigned int i = 0; i < parameters.GetSize(); ++i )
    {
      input >> parameters[ i ];
    }
  }
  else
  {
    std::cerr << "ERROR: could not open the text file containing the "
              << "parameter values." << std::endl;
    return 1;
  }
  transform->SetParameters( parameters );

  /** Declare variables. */
  InputPointType  inputPoint; inputPoint.Fill( 4.1 );
  OutputPointType outputPoint; double sum = 0.0;
  itk::TimeProbe  timeProbeOLD, timeProbeNEW;

  /** Time the TransformPoint with the old region iterator. */
  timeProbeOLD.Start();
  for( unsigned int i = 0; i < N; ++i )
  {
    outputPoint = transform->TransformPoint_OLD( inputPoint );
    sum        += outputPoint[ 0 ]; sum += outputPoint[ 1 ]; sum += outputPoint[ 2 ];
  }
  timeProbeOLD.Stop();
  const double oldTime = timeProbeOLD.GetMean();

  /** Time the TransformPoint with the new scanline iterator. */
  timeProbeNEW.Start();
  for( unsigned int i = 0; i < N; ++i )
  {
    outputPoint = transform->TransformPoint( inputPoint );
    sum        += outputPoint[ 0 ]; sum += outputPoint[ 1 ]; sum += outputPoint[ 2 ];
  }
  timeProbeNEW.Stop();
  const double newTime = timeProbeNEW.GetMean();

  // Avoid compiler optimizations, so use sum
  std::cerr << sum << std::endl; // works but ugly on screen
  //  volatile double a = sum; // works but gives unused variable warning
  //#pragma optimize( "", off ) // unrecognized pragma
  //sum += 2.0;

  /** Report timings. */
  std::cerr << std::setprecision( 4 );
  std::cerr << "Time OLD = " << oldTime << " " << timeProbeOLD.GetUnit() << std::endl;
  std::cerr << "Time NEW = " << newTime << " " << timeProbeNEW.GetUnit() << std::endl;
  std::cerr << "Speedup factor = " << oldTime / newTime << std::endl;

  /** Return a value. */
  return 0;

} // end main
