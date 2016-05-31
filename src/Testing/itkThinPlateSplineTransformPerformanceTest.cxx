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
#include "SplineKernelTransform/itkThinPlateSplineKernelTransform2.h"
#include "itkTransformixInputPointFileReader.h"

// Report timings
#include "itkTimeProbe.h"
#include "itkTimeProbesCollectorBase.h"

#include <fstream>
#include <iomanip>

#include "vnl/algo/vnl_qr.h"
//#include "vnl/algo/vnl_sparse_lu.h"
//#include "vnl/algo/vnl_cholesky.h"
#include "vnl/vnl_matlab_filewrite.h"
#include "vnl/vnl_matrix_fixed.h"
#include "vnl/vnl_sparse_matrix.h"

//-------------------------------------------------------------------------------------
// Helper class to be able to access protected functions and variables.

namespace itk
{

template< class TScalarType, unsigned int NDimensions >
class KernelTransformPublic :
  public ThinPlateSplineKernelTransform2< TScalarType, NDimensions >
{
public:

  typedef KernelTransformPublic Self;
  typedef ThinPlateSplineKernelTransform2<
    TScalarType, NDimensions >                Superclass;
  typedef SmartPointer< Self >       Pointer;
  typedef SmartPointer< const Self > ConstPointer;
  itkTypeMacro( KernelTransformPublic, ThinPlateSplineKernelTransform2 );
  itkNewMacro( Self );

  typedef typename Superclass::PointSetType    PointSetType;
  typedef typename Superclass::LMatrixType     LMatrixType;
  typedef typename Superclass::GMatrixType     GMatrixType;
  typedef typename Superclass::InputVectorType InputVectorType;

  void SetSourceLandmarksPublic( PointSetType * landmarks )
  {
    this->m_SourceLandmarks  = landmarks;
    this->m_WMatrixComputed  = false;
    this->m_LMatrixComputed  = false;
    this->m_LInverseComputed = false;
  }


  void ComputeLPublic( void )
  {
    this->ComputeL();
  }


  LMatrixType GetLMatrix( void ) const
  {
    return this->m_LMatrix;
  }


  void ComputeGPublic( const InputVectorType & landmarkVector,
    GMatrixType & GMatrix ) const
  {
    this->ComputeG( landmarkVector, GMatrix );
  }


};

// end helper class
} // end namespace itk

//-------------------------------------------------------------------------------------

// Test matrix inversion performance
// Test Jacobian computation performance
int
main( int argc, char * argv[] )
{
  /** Some basic type definitions. */
  const unsigned int Dimension = 3;
  // ScalarType double needed for Cholesky. Double is used in elastix.
  typedef double ScalarType;
  const unsigned long maxTestedLandmarksForSVD = 401;
  const ScalarType    tolerance                = 1e-8; // for double

  /** Check. */
  if( argc != 3 )
  {
    std::cerr << "ERROR: You should specify a text file with the thin plate spline "
              << "source (fixed image) landmarks." << std::endl;
    return 1;
  }

  /** Other typedefs. */
  typedef itk::KernelTransformPublic<
    ScalarType, Dimension >                             TransformType;
  typedef TransformType::JacobianType               JacobianType;
  typedef TransformType::NonZeroJacobianIndicesType NonZeroJacobianIndicesType;
  typedef TransformType::PointSetType               PointSetType;
  typedef itk::TransformixInputPointFileReader<
    PointSetType >                                      IPPReaderType;

  typedef PointSetType::PointsContainer PointsContainerType;
  typedef PointsContainerType::Pointer  PointsContainerPointer;
  typedef PointSetType::PointType       PointType;
  typedef TransformType::LMatrixType    LMatrixType;

  PointSetType::Pointer dummyLandmarks = PointSetType::New();

  /** Create the kernel transform. */
  TransformType::Pointer kernelTransform = TransformType::New();
  kernelTransform->SetStiffness( 0.0 ); // interpolating

  /** Read landmarks. */
  IPPReaderType::Pointer ippReader = IPPReaderType::New();
  ippReader->SetFileName( argv[ 1 ] );
  try
  {
    ippReader->Update();
  }
  catch( itk::ExceptionObject & excp )
  {
    std::cerr << "  Error while opening input point file." << std::endl;
    std::cerr << excp << std::endl;
    return 1;
  }

  // Expect points, not indices.
  if( ippReader->GetPointsAreIndices() )
  {
    std::cerr << "ERROR: landmarks should be specified as points (not indices)"
              << std::endl;
    return 1;
  }

  /** Get the set of input points. */
  PointSetType::Pointer sourceLandmarks = ippReader->GetOutput();
  //const unsigned long realNumberOfLandmarks = ippReader->GetNumberOfPoints();

  std::vector< unsigned long > usedNumberOfLandmarks;
  usedNumberOfLandmarks.push_back( 100 );
  usedNumberOfLandmarks.push_back( 200 );
//   usedNumberOfLandmarks.push_back( 500 );
//   usedNumberOfLandmarks.push_back( 1000 );
//   usedNumberOfLandmarks.push_back( realNumberOfLandmarks );

  std::cerr << "Matrix scalar type: "
            << typeid( ScalarType ).name() << "\n" << std::endl;

  // Loop over usedNumberOfLandmarks
  for( std::size_t i = 0; i < usedNumberOfLandmarks.size(); i++ )
  {
    itk::TimeProbesCollectorBase timeCollector;

    unsigned long numberOfLandmarks = usedNumberOfLandmarks[ i ];
    std::cerr << "----------------------------------------\n";
    std::cerr << "Number of specified landmarks: "
              << numberOfLandmarks << std::endl;

    /** Get subset. */
    PointsContainerPointer usedLandmarkPoints = PointsContainerType::New();
    PointSetType::Pointer  usedLandmarks      = PointSetType::New();
    for( unsigned long j = 0; j < numberOfLandmarks; j++ )
    {
      PointType tmp = ( *sourceLandmarks->GetPoints() )[ j ];
      usedLandmarkPoints->push_back( tmp );
    }
    usedLandmarks->SetPoints( usedLandmarkPoints );

    /** Set the ipp as source landmarks.
     * 1) Compute L matrix
     * 2) Compute inverse of L
     */

    LMatrixType lMatrixInverse1, lMatrixInverse2; //, lMatrixInverse4;

    /** Task 1: compute L. */
    timeCollector.Start( "ComputeL" );
    kernelTransform->SetSourceLandmarksPublic( usedLandmarks );
    kernelTransform->ComputeLPublic();
    LMatrixType lMatrix = kernelTransform->GetLMatrix();
    timeCollector.Stop( "ComputeL" );

    /** Task 2: Compute L inverse. */
    if( numberOfLandmarks < maxTestedLandmarksForSVD )
    {
      // Method 1: Singular Value Decomposition
      timeCollector.Start( "ComputeLInverseBySVD" );
      lMatrixInverse1 = vnl_svd< ScalarType >( lMatrix ).inverse();
      timeCollector.Stop( "ComputeLInverseBySVD" );
    }
    else
    {
      std::cerr << "L matrix inversion (method 1, svd) took: too long" << std::endl;
    }

    // Method 2: QR Decomposition
    timeCollector.Start( "ComputeLInverseByQR" );
    lMatrixInverse2 = vnl_qr< ScalarType >( lMatrix ).inverse();
    timeCollector.Stop( "ComputeLInverseByQR" );

    // Method 3: Cholesky decomposition
    // Cholesky decomposition does not work due to lMatrix not being positive definite.
    //   startClock = clock();
    //   LMatrixType lMatrixInverse3 = vnl_cholesky( lMatrix,
    //     vnl_cholesky::Operation::estimate_condition ).inverse();
    //   std::cerr << "L matrix inversion (method 3, cholesky ) took: "
    //     << clock() - startClock << " ms." << std::endl;

    /** The following code is out-commented.
     * It is used to test LU decomposition, which in vnl is only implemented
     * for sparse matrices. It also depends on a local modification of the
     * vnl_sparse_lu claas, where a method invert() was implemented similar
     * to the invert() of vnl_qr.inverse().
     */
//     // Convert to sparse matrix
//     startClock = clock();
//     LSparseMatrixType lSparseMatrix( lMatrix.rows(), lMatrix.cols() );
//     for ( unsigned int r = 0; r < lMatrix.rows(); r++ )
//     {
//       for ( unsigned int c = 0; c < lMatrix.cols(); c++ )
//       {
//         ScalarType val = lMatrix.get( r, c );
//         if ( val != 0 )
//         {
//           lSparseMatrix( r, c ) = val;
//         }
//       }
//     }
//     std::cerr << "Conversion to sparse matrix took: "
//       << clock() - startClock << " ms." << std::endl;
//
//     // Method 4: LU Decomposition
//     // Depends on local ITK vnl_sparse_lu modification
//     startClock = clock();
//     lMatrixInverse4 = vnl_sparse_lu( lSparseMatrix ).inverse();
//     std::cerr << "L matrix inversion (method 4,  lu) took: "
//       << clock() - startClock << " ms." << std::endl;

    /** Compute error compared to SVD. */
    if( numberOfLandmarks < maxTestedLandmarksForSVD )
    {
      double diff_qr = ( lMatrixInverse1 - lMatrixInverse2 ).frobenius_norm();
      //double diff_lu = (lMatrixInverse1a - lMatrixInverse4).frobenius_norm();

      std::cerr << "Frobenius difference of method 2 with SVD: "
                << diff_qr << std::endl;
      //std::cerr << "Frobenius difference of method 4 with SVD: " << diff_lu << std::endl;

      if( diff_qr > tolerance )
      {
        std::cerr
          << "ERROR: Frobenius difference of matrix inversion methods too big: "
          << diff_qr << std::endl;
        return 1;
      }
    }
    else
    {
      std::cerr << "Frobenius difference of method 2,4 with SVD: unknown" << std::endl;
    }

    //   startClock = clock();
    //   LMatrixType lMatrixInverse3 = vnl_lu<ScalarType>( kernelTransform->GetLMatrix() ).inverse();
    //   std::cerr << "L matrix inversion (method 2, lu ) took: "
    //     << clock() - startClock << " ms." << std::endl;

    // To do: Add SuiteSparse tests.

    // Write L Matrix to Matlab file. For inspection of matrix appearance.
    std::ostringstream makeFileName( "" );
    makeFileName << argv[ 2 ]
                 << "/LMatrix_N"
                 << numberOfLandmarks << ".mat";
    vnl_matlab_filewrite matlabWriter( makeFileName.str().c_str() );
    matlabWriter.write( lMatrix, "lMatrix" );
    matlabWriter.write( lMatrixInverse2, "lMatrixInverseQR" );

    //
    // Test Jacobian computation performance

    typedef vnl_matrix_fixed< ScalarType, Dimension, Dimension > GMatrixType;
    GMatrixType Gmatrix; // dim x dim
    typedef PointSetType::PointsContainerIterator PointsIterator;

    // OLD way:
    PointType p; p[ 0 ] = 10.0; p[ 1 ] = 13.0; p[ 2 ] = 11.0;
    timeCollector.Start( "ComputeJacobianOLD" );
    JacobianType jac1;
    jac1.SetSize( Dimension, numberOfLandmarks * Dimension );
    jac1.Fill( 0.0 );
    PointsIterator sp = usedLandmarks->GetPoints()->Begin();
    for( unsigned int lnd = 0; lnd < numberOfLandmarks; lnd++ )
    {
      kernelTransform->ComputeGPublic( p - sp->Value(), Gmatrix );
      for( unsigned int dim = 0; dim < Dimension; dim++ )
      {
        for( unsigned int odim = 0; odim < Dimension; odim++ )
        {
          for( unsigned int lidx = 0; lidx < numberOfLandmarks * Dimension; lidx++ )
          {
            jac1[ odim ][ lidx ] += Gmatrix( dim, odim )
              * lMatrixInverse2[ lnd * Dimension + dim ][ lidx ];
          }
        }
      }
      ++sp;
    }

    for( unsigned int odim = 0; odim < Dimension; odim++ )
    {
      for( unsigned long lidx = 0; lidx < numberOfLandmarks * Dimension; lidx++ )
      {
        for( unsigned int dim = 0; dim < Dimension; dim++ )
        {
          jac1[ odim ][ lidx ] += p[ dim ]
            * lMatrixInverse2[ ( numberOfLandmarks + dim ) * Dimension + odim ][ lidx ];
        }
        const unsigned long index = ( numberOfLandmarks + Dimension ) * Dimension + odim;
        jac1[ odim ][ lidx ] += lMatrixInverse2[ index ][ lidx ];
      }
    }
    timeCollector.Stop( "ComputeJacobianOLD" );

    // NEW way:

    /** Reset source landmarks, otherwise L is not recomputed. */
    kernelTransform->SetSourceLandmarks( dummyLandmarks );
    kernelTransform->SetSourceLandmarks( usedLandmarks );
    timeCollector.Start( "ComputeJacobianNEW" );
    JacobianType               jac2;
    NonZeroJacobianIndicesType nzji;
    kernelTransform->GetJacobian( p, jac2, nzji );
    timeCollector.Stop( "ComputeJacobianNEW" );

    // diff
    double diff_jac = ( jac1 - jac2 ).frobenius_norm();
    std::cerr << "Frobenius difference of jacs: " << diff_jac << std::endl;
    if( diff_jac > tolerance )
    {
      std::cerr << "ERROR: Frobenius difference of Jacobian computation too big: " << diff_jac << std::endl;
      return 1;
    }

    // Report timings
    timeCollector.Report();
    std::cout << std::endl;

  } // end loop

  /** Return a value. */
  return 0;

} // end main
