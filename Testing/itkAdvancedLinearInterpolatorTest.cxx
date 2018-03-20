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
/** \file
 \brief Compare the advanced linear interpolator with the linear and 1st order B-spline.
 */

#include "itkLinearInterpolateImageFunction.h"
#include "itkAdvancedLinearInterpolateImageFunction.h"
#include "itkBSplineInterpolateImageFunction.h"

#include "itkImage.h"
#include "itkImageRegionIterator.h"
#include "itkMersenneTwisterRandomVariateGenerator.h"
#include "itkImageFileWriter.h"
#include "vnl/vnl_math.h"
#include "itkTimeProbe.h"

//-------------------------------------------------------------------------------------

// Test function templated over the dimension
template< unsigned int Dimension >
bool
TestInterpolators( void )
{
  typedef itk::Image< short, Dimension >         InputImageType;
  typedef typename InputImageType::SizeType      SizeType;
  typedef typename InputImageType::SpacingType   SpacingType;
  typedef typename InputImageType::PointType     OriginType;
  typedef typename InputImageType::RegionType    RegionType;
  //typedef typename RegionType::IndexType         IndexType;
  typedef typename InputImageType::DirectionType DirectionType;
  typedef double                                 CoordRepType;
  typedef double                                 CoefficientType;

  typedef itk::LinearInterpolateImageFunction<
    InputImageType, CoordRepType >                  LinearInterpolatorType;
  typedef itk::AdvancedLinearInterpolateImageFunction<
    InputImageType, CoordRepType >                  AdvancedLinearInterpolatorType;
  typedef itk::BSplineInterpolateImageFunction<
    InputImageType, CoordRepType, CoefficientType > BSplineInterpolatorType;
  typedef typename LinearInterpolatorType::ContinuousIndexType         ContinuousIndexType;
  typedef typename AdvancedLinearInterpolatorType::CovariantVectorType CovariantVectorType;
  typedef typename AdvancedLinearInterpolatorType::OutputType          OutputType;  // double scalar

  typedef itk::ImageRegionIterator< InputImageType >             IteratorType;
  typedef itk::Statistics::MersenneTwisterRandomVariateGenerator RandomNumberGeneratorType;
  //typedef itk::ImageFileWriter< InputImageType >                 WriterType;

  RandomNumberGeneratorType::Pointer randomNum = RandomNumberGeneratorType::GetInstance();

  /** Create random input image. */
  SizeType size; SpacingType spacing; OriginType origin;
  for( unsigned int i = 0; i < Dimension; ++i )
  {
    size[ i ]    = 10;
    spacing[ i ] = randomNum->GetUniformVariate( 0.5, 2.0 );
    origin[ i ]  = randomNum->GetUniformVariate( -1, 0 );
  }
  RegionType region; region.SetSize( size );

  /** Make sure to test for non-identity direction cosines. */
  DirectionType direction; direction.Fill( 0.0 );
  if( Dimension == 2 )
  {
    direction[ 0 ][ 1 ] = -1.0;
    direction[ 1 ][ 0 ] =  1.0;
  }
  else if( Dimension == 3 )
  {
    direction[ 0 ][ 2 ] = -1.0;
    direction[ 1 ][ 1 ] =  1.0;
    direction[ 2 ][ 0 ] =  1.0;
  }

  typename InputImageType::Pointer image = InputImageType::New();
  image->SetRegions( region );
  image->SetOrigin( origin );
  image->SetSpacing( spacing );
  image->SetDirection( direction );
  image->Allocate();

  // loop over image and fill with random values
  IteratorType it( image, image->GetLargestPossibleRegion() );
  it.GoToBegin();

  while( !it.IsAtEnd() )
  {
    it.Set( randomNum->GetUniformVariate( 0, 255 ) );
    ++it;
  }

  /** Write the image. */
  //WriterType::Pointer writer = WriterType::New();
  //writer->SetInput( image );
  //writer->SetFileName( "image.mhd" );
  //writer->Update();

  /** Create and setup interpolators. */
  typename LinearInterpolatorType::Pointer linear          = LinearInterpolatorType::New();
  typename AdvancedLinearInterpolatorType::Pointer linearA = AdvancedLinearInterpolatorType::New();
  typename BSplineInterpolatorType::Pointer bspline        = BSplineInterpolatorType::New();
  linear->SetInputImage( image );
  linearA->SetInputImage( image );
  bspline->SetSplineOrder( 1 ); // prior to SetInputImage()
  bspline->SetInputImage( image );

  /** Test some points. */
  const unsigned int count = 12;
  double             darray1[ 12 ][ Dimension ];
  if( Dimension == 2 )
  {
    double darray2[ 12 ][ 2 ] =
    { { 0.1, 0.2 }, { 3.4, 5.8 }, { 4.0, 6.0 }, { 2.1, 8.0 },
      { -0.1, -0.1 }, { 0.0, 0.0 }, { 1.3, 1.0 }, { 2.0, 5.7 },
      { 9.5, 9.1 }, { 2.0, -0.1 }, { -0.1, 2.0 }, { 12.7, 15.3 } };
    for( unsigned int i = 0; i < 12; i++ )
    {
      for( unsigned int j = 0; j < Dimension; j++ )
      {
        darray1[ i ][ j ] = darray2[ i ][ j ];
      }
    }
  }
  else if( Dimension == 3 )
  {
    //double darray2[count][3] =
    //{ { 0.0, 0.0, 0.0}, { 0.1, 0.0, 0.0}, { 0.2, 0.0, 0.0} }; // x, y=z=0, works
    //{ { 0.0, 0.5, 0.0}, { 0.1, 0.5, 0.0}, { 0.2, 0.5, 0.0} }; // x, z=0, works
    //{ { 0.0, 0.0, 0.5}, { 0.1, 0.0, 0.5}, { 0.2, 0.0, 0.5} }; // x, y=0, works
    //{ { 0.0, 0.2, 0.2}, { 0.0, 0.4, 0.4}, { 0.0, 0.5, 0.5} }; // x=0, y=z, works
    //{ { 0.0, 0.0, 0.0}, { 0.0, 0.1, 0.0}, { 0.0, 0.2, 0.0} }; // y, works
    //{ { 0.0, 0.0, 0.0}, { 0.0, 0.0, 0.1}, { 0.0, 0.0, 0.2} }; // z, works
    //{ { 0.0, 0.0, 0.0}, { 0.2, 0.1, 0.0}, { 0.5, 0.2, 0.0} }; // xy, works
    //{ { 0.0, 0.0, 0.0}, { 0.3, 0.0, 0.1}, { 0.5, 0.0, 0.2} }; // xz, works
    //{ { 0.0, 0.0, 0.0}, { 0.0, 0.1, 0.1}, { 0.0, 0.4, 0.2} }; // yz, works
    double darray2[ 12 ][ 3 ] =
    { { 0.1, 0.2, 0.1 }, { 3.4, 5.8, 4.7 }, { 4.0, 6.0, 5.0 }, { 2.1, 8.0, 3.4 },
      { -0.1, -0.1, -0.1 }, { 0.0, 0.0, 0.0 }, { 1.3, 1.0, 1.4 }, { 2.0, 5.7, 7.5 },
      { 9.5, 9.1, 9.3 }, { 2.0, -0.1, 5.3 }, { -0.1, 2.0, 4.0 }, { 12.7, 15.3, 14.1 } };
    for( unsigned int i = 0; i < count; i++ )
    {
      for( unsigned int j = 0; j < Dimension; j++ )
      {
        darray1[ i ][ j ] = darray2[ i ][ j ];
      }
    }
  }

  /** Compare results. */
  OutputType          valueLinA, valueBSpline, valueBSpline2;
  CovariantVectorType derivLinA, derivBSpline, derivBSpline2;
  for( unsigned int i = 0; i < count; i++ )
  {
    ContinuousIndexType cindex( &darray1[ i ][ 0 ] );

    linearA->EvaluateValueAndDerivativeAtContinuousIndex( cindex, valueLinA, derivLinA );
    valueBSpline = bspline->EvaluateAtContinuousIndex( cindex );
    derivBSpline = bspline->EvaluateDerivativeAtContinuousIndex( cindex );
    bspline->EvaluateValueAndDerivativeAtContinuousIndex( cindex, valueBSpline2, derivBSpline2 );

    std::cout << "cindex: " << cindex << std::endl;

    if (linear->IsInsideBuffer(cindex))
    {
      std::cout << "linear:   " << linear->EvaluateAtContinuousIndex(cindex) << "   ---" << std::endl;
    }
    else
    {
      std::cout << "linear:   ---    ---" << std::endl;
    }
    std::cout << "linearA:  " << valueLinA     << "   " << derivLinA     << std::endl;
    std::cout << "B-spline: " << valueBSpline  << "   " << derivBSpline  << std::endl;
    std::cout << "B-spline: " << valueBSpline2 << "   " << derivBSpline2 << "\n" << std::endl;

    if( vnl_math_abs( valueLinA - valueBSpline ) > 1.0e-3 )
    {
      std::cerr << "ERROR: there is a difference in the interpolated value, "
                << "between the linear and the 1st-order B-spline interpolator." << std::endl;
      return false;
    }
    if( vnl_math_abs( valueBSpline - valueBSpline2 ) > 1.0e-3 )
    {
      std::cerr << "ERROR: there is a difference in the interpolated value, "
                << "within the 1st-order B-spline interpolator (inconsistency)." << std::endl;
      return false;
    }
    if( ( derivLinA - derivBSpline ).GetVnlVector().magnitude() > 1.0e-3 )
    {
      std::cerr << "ERROR: there is a difference in the interpolated gradient, "
                << "between the linear and the 1st-order B-spline interpolator." << std::endl;
      return false;
    }
    if( ( derivBSpline - derivBSpline2 ).GetVnlVector().magnitude() > 1.0e-3 )
    {
      std::cerr << "ERROR: there is a difference in the interpolated gradient, "
                << "within the 1st-order B-spline interpolator (inconsistency)." << std::endl;
      return false;
    }
  }

  /** Measure the run times, but only in release mode. */
#ifdef NDEBUG
  std::cout << std::endl;
  ContinuousIndexType cindex( &darray1[ 1 ][ 0 ] );
  std::cout << "cindex: " << cindex << std::endl;
  OutputType         value; CovariantVectorType deriv;
  const unsigned int runs = 1e5;

  itk::TimeProbe timer;
  timer.Start();
  for( unsigned int i = 0; i < runs; ++i )
  {
    value = linear->EvaluateAtContinuousIndex( cindex );
  }
  timer.Stop();
  std::cout << "linear  (value) : "
            << 1.0e3 * timer.GetMean() / static_cast< double >( runs )
            << " ms" << std::endl;

  timer.Reset(); timer.Start();
  for( unsigned int i = 0; i < runs; ++i )
  {
    linearA->EvaluateValueAndDerivativeAtContinuousIndex( cindex, value, deriv );
  }
  timer.Stop();
  std::cout << "linearA (v&d)   : "
            << 1.0e3 * timer.GetMean() / static_cast< double >( runs )
            << " ms" << std::endl;

  timer.Reset(); timer.Start();
  for( unsigned int i = 0; i < runs; ++i )
  {
    value = bspline->EvaluateAtContinuousIndex( cindex );
  }
  timer.Stop();
  std::cout << "B-spline (value): "
            << 1.0e3 * timer.GetMean() / static_cast< double >( runs )
            << " ms" << std::endl;

  timer.Reset(); timer.Start();
  for( unsigned int i = 0; i < runs; ++i )
  {
    value = bspline->EvaluateAtContinuousIndex( cindex );
    deriv = bspline->EvaluateDerivativeAtContinuousIndex( cindex );
  }
  timer.Stop();
  std::cout << "B-spline (v+d)  : "
            << 1.0e3 * timer.GetMean() / static_cast< double >( runs )
            << " ms" << std::endl;

  timer.Reset(); timer.Start();
  for( unsigned int i = 0; i < runs; ++i )
  {
    bspline->EvaluateValueAndDerivativeAtContinuousIndex( cindex, value, deriv );
  }
  timer.Stop();
  std::cout << "B-spline (v&d)  : "
            << 1.0e3 * timer.GetMean() / static_cast< double >( runs )
            << " ms" << std::endl;
#endif

  return true;

} // end TestInterpolator()


int
main( int argc, char ** argv )
{
  // 2D tests
  bool success = TestInterpolators< 2 >();
  if( !success ) { return EXIT_FAILURE; }

  std::cerr << "\n\n\n-----------------------------------\n\n\n";

  // 3D tests
  success = TestInterpolators< 3 >();
  if( !success ) { return EXIT_FAILURE; }

  return EXIT_SUCCESS;
} // end main
