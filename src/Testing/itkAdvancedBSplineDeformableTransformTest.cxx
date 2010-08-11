
#include "itkAdvancedBSplineDeformableTransform.h"
#include "itkGridScheduleComputer.h"

#include <ctime>
#include <fstream>
#include <iomanip>

//-------------------------------------------------------------------------------------

int main( int argc, char *argv[] )
{
  /** Some basic type definitions.
   * NOTE: don't change the dimension or the spline order, since the
   * hard-coded ground truth depends on this.
   */
  const unsigned int Dimension = 3;
  const unsigned int SplineOrder = 3;
  typedef float CoordinateRepresentationType;
  //const double distance = 1e-3; // the allowable distance
  //const double allowedTimeDifference = 0.1; // 10% is considered within limits
  /** The number of calls to Evaluate(). This number gives reasonably
   * fast test results in Release mode.
   */
  unsigned int N = static_cast<unsigned int>( 1e5 );

  /** Check. */
  if ( argc != 2 )
  {
    std::cerr << "ERROR: You should specify a text file with the B-spline "
      << "transformation parameters." << std::endl;
    return 1;
  }

  /** Other typedefs. */
  typedef itk::AdvancedBSplineDeformableTransform<
    CoordinateRepresentationType, Dimension, SplineOrder >    TransformType;
  typedef TransformType::JacobianType                   JacobianType;
  typedef TransformType::SpatialJacobianType            SpatialJacobianType;
  typedef TransformType::SpatialHessianType             SpatialHessianType;
  typedef TransformType::JacobianOfSpatialJacobianType  JacobianOfSpatialJacobianType;
  typedef TransformType::JacobianOfSpatialHessianType   JacobianOfSpatialHessianType;
  typedef TransformType::NonZeroJacobianIndicesType     NonZeroJacobianIndicesType;
  typedef TransformType::InputPointType                 InputPointType;
  typedef TransformType::ParametersType                 ParametersType;
  typedef itk::Image< CoordinateRepresentationType,
    Dimension >                                         InputImageType;
  typedef InputImageType::RegionType    RegionType;
  typedef InputImageType::SizeType      SizeType;
  typedef InputImageType::IndexType     IndexType;
  typedef InputImageType::SpacingType   SpacingType;
  typedef InputImageType::PointType     OriginType;

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

  transform->SetGridOrigin( gridOrigin );
  transform->SetGridSpacing( gridSpacing );
  transform->SetGridRegion( gridRegion );

  /** Now read the parameters as defined in the file par.txt. */
  ParametersType parameters( transform->GetNumberOfParameters() );
  //std::ifstream input( "D:/toolkits/elastix/src/Testing/par.txt" );
  std::ifstream input( argv[ 1 ] );
  if ( input.is_open() )
  {
    for ( unsigned int i = 0; i < parameters.GetSize(); ++i )
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

  /** Get the number of nonzero Jacobian indices. */
  unsigned long nonzji = transform->GetNumberOfNonZeroJacobianIndices();

  /** Declare variables. */
  InputPointType inputPoint;
  inputPoint.Fill( 4.1 );
  JacobianType jacobian;
  SpatialJacobianType spatialJacobian;
  SpatialHessianType spatialHessian;
  JacobianOfSpatialJacobianType jacobianOfSpatialJacobian;
  JacobianOfSpatialHessianType jacobianOfSpatialHessian;
  NonZeroJacobianIndicesType nzji;

  /** Resize some of the variables. */
  nzji.resize( nonzji );
  jacobian.SetSize( Dimension, nonzji );
  jacobianOfSpatialJacobian.resize( nonzji );
  jacobianOfSpatialHessian.resize( nonzji );

  /**
   *
   * Call functions for testing that they don't crash.
   *
   */

  /** The Jacobian. */
  transform->GetJacobian( inputPoint, jacobian, nzji );

  /** The spatial Jacobian. */
  transform->GetSpatialJacobian( inputPoint, spatialJacobian );

  /** The spatial Hessian. */
  transform->GetSpatialHessian( inputPoint, spatialHessian );

  /** The Jacobian of the spatial Jacobian. */
  transform->GetJacobianOfSpatialJacobian( inputPoint,
    jacobianOfSpatialJacobian, nzji );

  transform->GetJacobianOfSpatialJacobian( inputPoint,
    spatialJacobian, jacobianOfSpatialJacobian, nzji );

  /** The Jacobian of the spatial Hessian. */
  transform->GetJacobianOfSpatialHessian( inputPoint,
    jacobianOfSpatialHessian, nzji );

  transform->GetJacobianOfSpatialHessian( inputPoint,
    spatialHessian, jacobianOfSpatialHessian, nzji );

//   /***/
//   transform->GetSpatialHessian( inputPoint, spatialHessian );
//   for ( unsigned int i = 0; i < Dimension; ++i )
//   {
//     std::cerr << spatialHessian[ i ] << std::endl;
//   }
//
//   /***/
//   transform->GetJacobianOfSpatialHessian( inputPoint,
//     spatialHessian, jacobianOfSpatialHessian, nzji );
//   for ( unsigned int mu = 0; mu < 2; ++mu )
//   {
//     for ( unsigned int i = 0; i < Dimension; ++i )
//     {
//       std::cerr << jacobianOfSpatialHessian[ mu ][ i ] << std::endl;
//     }
//   }

  /**
   *
   * Call functions for timing.
   *
   */

  /** Time the implementation of the spatial Jacobian. */
  clock_t startClock = clock();
  for ( unsigned int i = 0; i < N; ++i )
  {
    transform->GetSpatialJacobian( inputPoint, spatialJacobian );
  }
  clock_t endClock = clock();
  clock_t clockITK = endClock - startClock;
  std::cerr << "The elapsed time for the spatial Jacobian is: "
    << clockITK << std::endl;

  /** Time the implementation of the spatial Hessian. */
  startClock = clock();
  for ( unsigned int i = 0; i < N; ++i )
  {
    transform->GetSpatialHessian( inputPoint, spatialHessian );
  }
  endClock = clock();
  clockITK = endClock - startClock;
  std::cerr << "The elapsed time for the spatial Hessian is: "
    << clockITK << std::endl;

  /** Time the implementation of the Jacobian of the spatial Jacobian. */
  startClock = clock();
  for ( unsigned int i = 0; i < N; ++i )
  {
    transform->GetJacobianOfSpatialJacobian( inputPoint,
      jacobianOfSpatialJacobian, nzji );
    //     transform->GetJacobianOfSpatialJacobian( inputPoint,
    //       spatialJacobian, jacobianOfSpatialJacobian, nzji );
  }
  endClock = clock();
  clockITK = endClock - startClock;
  std::cerr << "The elapsed time for the Jacobian of the spatial Jacobian is: "
    << clockITK << std::endl;

  /** Time the implementation of the Jacobian of the spatial Hessian. */
  startClock = clock();
  for ( unsigned int i = 0; i < N; ++i )
  {
    transform->GetJacobianOfSpatialHessian( inputPoint,
      jacobianOfSpatialHessian, nzji );
//     transform->GetJacobianOfSpatialHessian( inputPoint,
//       spatialHessian, jacobianOfSpatialHessian, nzji );
  }
  endClock = clock();
  clockITK = endClock - startClock;
  std::cerr << "The elapsed time for the Jacobian of the spatial Hessian is: "
    << clockITK << std::endl;

  /** Time the implementation of the spatial Jacobian and its Jacobian. */
  startClock = clock();
  for ( unsigned int i = 0; i < N; ++i )
  {
    transform->GetSpatialJacobian( inputPoint, spatialJacobian );
    transform->GetJacobianOfSpatialJacobian( inputPoint,
      jacobianOfSpatialJacobian, nzji );
  }
  endClock = clock();
  clockITK = endClock - startClock;
  std::cerr << "The elapsed time for the spatial Jacobian (2 func) is: "
    << clockITK << std::endl;

  /** Time the implementation of the spatial Jacobian and its Jacobian. */
  startClock = clock();
  for ( unsigned int i = 0; i < N; ++i )
  {
    transform->GetJacobianOfSpatialJacobian( inputPoint,
      spatialJacobian, jacobianOfSpatialJacobian, nzji );
  }
  endClock = clock();
  clockITK = endClock - startClock;
  std::cerr << "The elapsed time for the spatial Jacobian (1 func) is: "
    << clockITK << std::endl;

  /** Time the implementation of the spatial Hessian. */
  startClock = clock();
  for ( unsigned int i = 0; i < N; ++i )
  {
    transform->GetSpatialHessian( inputPoint, spatialHessian );
    transform->GetJacobianOfSpatialHessian( inputPoint,
      jacobianOfSpatialHessian, nzji );
  }
  endClock = clock();
  clockITK = endClock - startClock;
  std::cerr << "The elapsed time for the spatial Hessian (2 func) is: "
    << clockITK << std::endl;

  /** Time the implementation of the Jacobian of the spatial Hessian. */
  startClock = clock();
  for ( unsigned int i = 0; i < N; ++i )
  {
    transform->GetJacobianOfSpatialHessian( inputPoint,
      spatialHessian, jacobianOfSpatialHessian, nzji );
  }
  endClock = clock();
  clockITK = endClock - startClock;
  std::cerr << "The elapsed time for the spatial Hessian (1 func) is: "
    << clockITK << std::endl;

  /** Return a value. */
  return 0;

} // end main
