/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#include "SplineKernelTransform/itkThinPlateSplineKernelTransform2.h"
#include "itkTransformixInputPointFileReader.h"

#include <ctime>
#include <fstream>
#include <iomanip>

#include "vnl/algo/vnl_qr.h"
//#include "vnl/algo/vnl_sparse_lu.h"
//#include "vnl/algo/vnl_cholesky.h"

//-------------------------------------------------------------------------------------
// Helper class to be able to access protected functions and variables.

namespace itk {

template <class TScalarType, unsigned int NDimensions>
class KernelTransformPublic
  : public ThinPlateSplineKernelTransform2<TScalarType, NDimensions>
{
public:
  typedef KernelTransformPublic               Self;
  typedef ThinPlateSplineKernelTransform2<
    TScalarType, NDimensions >                Superclass;
  typedef SmartPointer<Self>                  Pointer;
  typedef SmartPointer<const Self>            ConstPointer;
  itkTypeMacro( KernelTransformPublic, ThinPlateSplineKernelTransform2 );
  itkNewMacro( Self );

  typedef typename Superclass::PointSetType   PointSetType;
  typedef typename Superclass::LMatrixType    LMatrixType;

  void SetSourceLandmarksPublic( PointSetType * landmarks )
  {
    this->m_SourceLandmarks = landmarks;
    this->m_WMatrixComputed = false;
    this->m_LMatrixComputed = false;
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
}; // end helper class
} // end namespace itk

//-------------------------------------------------------------------------------------

int main( int argc, char *argv[] )
{
  /** Some basic type definitions. */
  const unsigned int Dimension = 3;
  typedef double   ScalarType; // double needed for cholesky

  /** Check. */
  if ( argc != 2 )
  {
    std::cerr << "ERROR: You should specify a text file with the thin plate spline "
      << "source (fixed image) landmarks." << std::endl;
    return 1;
  }

  /** Other typedefs. */
  typedef itk::KernelTransformPublic<
    ScalarType, Dimension >                             TransformType;
  typedef TransformType::PointSetType                   PointSetType;
  typedef itk::TransformixInputPointFileReader<
    PointSetType >                                      IPPReaderType;

  typedef PointSetType::PointsContainer                 PointsContainerType;
  typedef PointsContainerType::Pointer                  PointsContainerPointer;
  typedef PointSetType::PointType                       PointType;
  typedef TransformType::LMatrixType                    LMatrixType;

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
  catch ( itk::ExceptionObject & excp )
  {
    std::cerr << "  Error while opening input point file." << std::endl;
    std::cerr << excp << std::endl;
    return 1;
  }

  // Expect points, not indices.
  if ( ippReader->GetPointsAreIndices() )
  {
    std::cerr << "ERROR: landmarks should be specified as points (not indices)"
      << std::endl;
    return 1;
  }

  /** Get the set of input points. */
  PointSetType::Pointer sourceLandmarks = ippReader->GetOutput();
  const unsigned long realNumberOfLandmarks = ippReader->GetNumberOfPoints();

  std::vector<unsigned long> usedNumberOfLandmarks;
  usedNumberOfLandmarks.push_back( 100 );
  usedNumberOfLandmarks.push_back( 200 );
  //usedNumberOfLandmarks.push_back( 500 );
  //usedNumberOfLandmarks.push_back( 1000 );
  //usedNumberOfLandmarks.push_back( realNumberOfLandmarks );

  std::cerr << "Matrix scalar type: " << typeid( ScalarType ).name()
    << std::endl;

  // Loop over usedNumberOfLandmarks
  for ( std::size_t i = 0; i < usedNumberOfLandmarks.size(); i++ )
  {
    unsigned long numberOfLandmarks = usedNumberOfLandmarks[ i ];
    std::cerr << "----------------------------------------\n";
    std::cerr << "Number of specified landmarks: "
      << numberOfLandmarks << std::endl;

    /** Get subset. */
    PointsContainerPointer usedLandmarkPoints = PointsContainerType::New();
    PointSetType::Pointer usedLandmarks = PointSetType::New();
    for ( unsigned long j = 0; j < numberOfLandmarks; j++ )
    {
      PointType tmp = (*sourceLandmarks->GetPoints())[ j ];
      usedLandmarkPoints->push_back( tmp );
    }
    usedLandmarks->SetPoints( usedLandmarkPoints );

    /** Set the ipp as source landmarks.
     * 1) Compute L matrix
     * 2) Compute inverse of L
     */

    LMatrixType lMatrixInverse1a, lMatrixInverse2;

    /** Task 1: compute L. */
    clock_t startClock = clock();
    kernelTransform->SetSourceLandmarksPublic( usedLandmarks );
    kernelTransform->ComputeLPublic();
    std::cerr << "Computing L matrix took "
      << clock() - startClock << " ms." << std::endl;

    /** Task 2: Compute L inverse. */
    if ( numberOfLandmarks < 1001 )
    {
      // Method 1: Singular Value Decomposition
      startClock = clock();
      lMatrixInverse1a = vnl_matrix_inverse<ScalarType>( kernelTransform->GetLMatrix() );
      std::cerr << "L matrix inversion (method 1, svd) took: "
        << clock() - startClock << " ms." << std::endl;
    }
    else
    {
      std::cerr << "L matrix inversion (method 1, svd) took: too long" << std::endl;
    }

    // Method 1b: Singular Value Decomposition
    //   startClock = clock();
    //   LMatrixType lMatrixInverse1b = vnl_svd<ScalarType>( kernelTransform->GetLMatrix() ).inverse();
    //   std::cerr << "L matrix inversion (method 1b, svd) took: "
    //     << clock() - startClock << " ms." << std::endl;

    // Method 2: QR Decomposition
    startClock = clock();
    lMatrixInverse2 = vnl_qr<ScalarType>( kernelTransform->GetLMatrix() ).inverse();
    std::cerr << "L matrix inversion (method 2,  qr ) took: "
      << clock() - startClock << " ms." << std::endl;

    // Method 3: Cholesky decomposition
    // Cholesky decomposition does not seem to work due to not pos. def.
    //   startClock = clock();
    //   LMatrixType lMatrixInverse3 = vnl_cholesky( kernelTransform->GetLMatrix(), vnl_cholesky::Operation::estimate_condition ).inverse();
    //   std::cerr << "L matrix inversion (method 3, cholesky ) took: "
    //     << clock() - startClock << " ms." << std::endl;

    //   std::cerr << "rows, cols = " << lMatrixInverse3.rows() << ", "
    //     << lMatrixInverse3.cols() << std::endl;


    /** Compute error compared to SVD. */
    if ( numberOfLandmarks < 1001 )
    {
      double diff_qr = (lMatrixInverse1a - lMatrixInverse2).frobenius_norm();
      //double diff_ch = (lMatrixInverse1a - lMatrixInverse3).frobenius_norm();

      std::cerr << "Frobenius difference of method 2 with SVD: " << diff_qr << std::endl;
      //std::cerr << "Frobenius difference of method 3 with SVD: " << diff_ch << std::endl;
    }
    else
    {
      std::cerr << "Frobenius difference of method 2 with SVD: unknown" << std::endl;
    }

    //   startClock = clock();
    //   LMatrixType lMatrixInverse3 = vnl_lu<ScalarType>( kernelTransform->GetLMatrix() ).inverse();
    //   std::cerr << "L matrix inversion (method 2, lu ) took: "
    //     << clock() - startClock << " ms." << std::endl;

    // Add SuiteSparse stuff.

  } // end loop

  /** Return a value. */
  return 0;

} // end main
