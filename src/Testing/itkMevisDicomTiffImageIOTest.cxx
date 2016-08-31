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

#include <iomanip>

#include "itkUseMevisDicomTiff.h"

#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImageRegionIterator.h"
#include "itkImageIOBase.h"
#include <string>
#include "itkMath.h"
#include "itkTestingComparisonImageFilter.h"

//-------------------------------------------------------------------------------------
// This test tests the itkMevisDicomTiffImageIO library. The test is performed
// in 2D, 3D, and 4D, for a unsigned char image. An artificial image is generated,
// written to disk, read from disk, and compared to the original.

template< unsigned int Dimension >
int
testMevis( void )
{
  std::cerr << "Testing write/read of " << Dimension << "D image..." << std::endl;

  /** Some basic type definitions. */
  typedef unsigned char PixelType;

  typedef itk::Image< PixelType, Dimension > ImageType;
  typedef itk::ImageFileWriter< ImageType >  WriterType;
  typedef itk::ImageFileReader< ImageType >  ReaderType;
  typedef itk::Testing::ComparisonImageFilter<
    ImageType, ImageType >                    ComparisonFilterType;
  typedef typename ImageType::SizeType          SizeType;
  typedef typename ImageType::SpacingType       SpacingType;
  typedef typename ImageType::PointType         OriginType;
  typedef typename ImageType::DirectionType     DirectionType;
  typedef itk::ImageRegionIterator< ImageType > IteratorType;

  typename WriterType::Pointer writer    = WriterType::New();
  typename ReaderType::Pointer reader    = ReaderType::New();
  typename ImageType::Pointer inputImage = ImageType::New();
  SizeType      size;
  SpacingType   spacing;
  OriginType    origin;
  DirectionType direction;

  direction.Fill( 0.0 );
  for( unsigned int i = 0; i < Dimension; ++i )
  {
    size[ i ]           = 20 + i;
    spacing[ i ]        = 0.5 + 0.1 * i;
    origin[ i ]         = 5 + 3 * i;
    direction[ i ][ i ] = 1.0; // default, will be changed below
  }
  // 4th dimension origin/spacing are lost using dcm/tiff format!
  if( Dimension == 4 )
  {
    // default values
    spacing[ 3 ] = 1.0;
    origin[ 3 ]  = 0.0;
  }

  // Difficult direction cosines:
  if( Dimension == 2 )
  {
    // Test flips
    direction[ 0 ][ 0 ] = 1; direction[ 0 ][ 1 ] = 0;
    direction[ 1 ][ 0 ] = 0; direction[ 1 ][ 1 ] = -1;
  }
  else if( Dimension == 3 )
  {
    // Test axis permutations
    // RM: won't work, because dicom always assume right hand
    // coordinate system when reconstructing the third direction
    /*
    direction[0][0] = 1; direction[0][1] = 0; direction[0][2] = 0;
    direction[1][0] = 0; direction[1][1] = 0; direction[1][2] = 1;
    direction[2][0] = 0; direction[2][1] = 1; direction[2][2] = 0;
    */

    // this works:
    direction[ 0 ][ 0 ] = 1; direction[ 0 ][ 1 ] = 0; direction[ 0 ][ 2 ] = 0;
    direction[ 1 ][ 0 ] = 0; direction[ 1 ][ 1 ] = 0; direction[ 1 ][ 2 ] = -1;
    direction[ 2 ][ 0 ] = 0; direction[ 2 ][ 1 ] = 1; direction[ 2 ][ 2 ] = 0;

  }
  else if( Dimension == 4 )
  {
    // Test 4D. RM: also make sure it is a right hand coordinate system
    direction[ 0 ][ 0 ] = 1; direction[ 0 ][ 1 ] = 0; direction[ 0 ][ 2 ] = 0; direction[ 0 ][ 3 ] = 0;
    direction[ 1 ][ 0 ] = 0; direction[ 1 ][ 1 ] = 0; direction[ 1 ][ 2 ] = -1; direction[ 1 ][ 3 ] = 0;
    direction[ 2 ][ 0 ] = 0; direction[ 2 ][ 1 ] = 1; direction[ 2 ][ 2 ] = 0; direction[ 2 ][ 3 ] = 0;
    direction[ 3 ][ 0 ] = 0; direction[ 3 ][ 1 ] = 0; direction[ 3 ][ 2 ] = 0; direction[ 3 ][ 3 ] = 1;
  }

  /** Generate image with gradually increasing pixel values. */
  inputImage->SetRegions( size );
  inputImage->SetSpacing( spacing );
  inputImage->SetOrigin( origin );
  inputImage->SetDirection( direction );

  try
  {
    inputImage->Allocate();
  }
  catch( itk::ExceptionObject & err )
  {
    std::cerr << "ERROR: Failed to allocate test image" << std::endl;
    std::cerr << err << std::endl;
    return 1;
  }

  IteratorType  it( inputImage, inputImage->GetLargestPossibleRegion() );
  unsigned long nrPixels = inputImage->GetLargestPossibleRegion().GetNumberOfPixels();
  double        factor   = itk::NumericTraits< PixelType >::max() / static_cast< double >( nrPixels );

  PixelType     pixval = 0;
  unsigned long pixnr  = 0;
  for( it.GoToBegin(); !it.IsAtEnd(); ++it )
  {
    pixval = static_cast< PixelType >(
      itk::Math::Floor< double >( pixnr * factor ) );
    it.Set( pixval );
    ++pixnr;
  }

  std::string testfile( "testimageMevisDicomTiff.tif" );
  writer->SetFileName( testfile );
  writer->SetInput( inputImage );
  reader->SetFileName( testfile );

  std::string task( "" );
  try
  {
    task = "Writing";
    writer->Update();
    task = "Reading";
    reader->Update();
  }
  catch( itk::ExceptionObject & err )
  {
    std::cerr << "ERROR: " << task << " mevis dicomtiff failed." << std::endl;
    std::cerr << err << std::endl;
    return 1;
  }

  std::cerr << "\nPrintSelf():\n" << std::endl;
  typename itk::ImageIOBase::Pointer mevisIO = reader->GetImageIO();
  mevisIO->Print( std::cerr, 2 );

  typename ImageType::Pointer outputImage = reader->GetOutput();
  bool same = true;
  same &= size == outputImage->GetLargestPossibleRegion().GetSize();
  same &= spacing == outputImage->GetSpacing();
  same &= origin == outputImage->GetOrigin();
  same &= direction == outputImage->GetDirection();

  if( !same )
  {
    std::cerr << "ERROR: image properties are not preserved" << std::endl;
    std::cerr << "Original properties:" << std::endl;
    inputImage->Print( std::cerr, 0 );
    std::cerr << "After write/read:" << std::endl;
    outputImage->Print( std::cerr, 0 );
    return 1;
  }

  typename ComparisonFilterType::Pointer comparisonFilter = ComparisonFilterType::New();
  comparisonFilter->SetTestInput( outputImage );
  comparisonFilter->SetValidInput( inputImage );
  comparisonFilter->Update();
  unsigned long nrDiffPixels = comparisonFilter->GetNumberOfPixelsWithDifferences();

  if( nrDiffPixels > 0 )
  {
    std::cerr << "ERROR: the pixel values are not correct after write/read" << std::endl;
    return 1;
  }

  return 0;

} // end templated function


int
main( int argc, char * argv[] )
{

  /** Support Mevis Dicom Tiff (if selected in cmake) */
  RegisterMevisDicomTiff();

#ifdef _ELASTIX_USE_MEVISDICOMTIFF

  /** Test for 2d, 3d, and 4d images */
  int ret2d = testMevis< 2 >();
  int ret3d = testMevis< 3 >();
  int ret4d = testMevis< 4 >();

  /** Return a value. */
  return ( ret2d | ret3d | ret4d );

#else

  std::cerr << "Elastix was not built with Mevis DicomTiff support, so this test is not relevant." << std::endl;
  return 0;

#endif

} // end main
