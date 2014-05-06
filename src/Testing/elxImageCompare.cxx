/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
/** \file
 \brief Compare two images.

 \verbinclude imagecompare.help
 */
#include "itkCommandLineArgumentParser.h"

#include "itkNumericTraits.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkExtractImageFilter.h"
#include "itksys/SystemTools.hxx"

#include "itkImageSource.h" // This should not be necessary after ITK patch is merged
#include "itkTestingComparisonImageFilter.h"

/**
 * ******************* GetHelpString *******************
 */

std::string
GetHelpString( void )
{
  std::stringstream ss;
  ss << "Usage:" << std::endl
     << "elxImageCompare" << std::endl
     << "  -test      image filename to test against baseline\n"
     << "  -base      baseline image filename\n"
     << "  [-t]       intensity difference threshold, default 0\n"
     << "  [-a]       allowable tolerance (# voxels different), default 0";
  return ss.str();

} // end GetHelpString()


// This comparison works on all image types by reading images in a 6D double images. If images > 6 dimensions
// must be compared, change this variable.
static const unsigned int ITK_TEST_DIMENSION_MAX = 4;

int
main( int argc, char ** argv )
{
  itk::CommandLineArgumentParser::Pointer parser = itk::CommandLineArgumentParser::New();
  parser->SetCommandLineArguments( argc, argv );
  parser->SetProgramHelpText( GetHelpString() );

  parser->MarkArgumentAsRequired( "-test", "The input filename." );
  parser->MarkArgumentAsRequired( "-base", "The baseline image filename." );

  itk::CommandLineArgumentParser::ReturnValue validateArguments = parser->CheckForRequiredArguments();

  if( validateArguments == itk::CommandLineArgumentParser::FAILED )
  {
    return EXIT_FAILURE;
  }
  else if( validateArguments == itk::CommandLineArgumentParser::HELPREQUESTED )
  {
    return EXIT_SUCCESS;
  }

  std::string testImageFileName;
  parser->GetCommandLineArgument( "-test", testImageFileName );

  std::string baselineImageFileName;
  parser->GetCommandLineArgument( "-base", baselineImageFileName );

  double diffThreshold = 0.0;
  parser->GetCommandLineArgument( "-t", diffThreshold );

  unsigned long allowedTolerance = 0;
  parser->GetCommandLineArgument( "-a", allowedTolerance );

  // Read images
  typedef itk::Image< double, ITK_TEST_DIMENSION_MAX > ImageType;
  typedef itk::ImageFileReader< ImageType >            ReaderType;

  // Read the baseline file
  ReaderType::Pointer baselineReader = ReaderType::New();
  baselineReader->SetFileName( baselineImageFileName );
  try
  {
    baselineReader->Update();
  }
  catch( itk::ExceptionObject & err )
  {
    std::cerr << "Error during reading baseline image: " << err << std::endl;
    return EXIT_FAILURE;
  }

  // Read the file to test
  ReaderType::Pointer testReader = ReaderType::New();
  testReader->SetFileName( testImageFileName );
  try
  {
    testReader->Update();
  }
  catch( itk::ExceptionObject & err )
  {
    std::cerr << "Error during reading test image: " << err << std::endl;
    return EXIT_FAILURE;
  }

  // The sizes of the baseline and test image must match
  ImageType::SizeType baselineSize;
  baselineSize = baselineReader->GetOutput()->GetLargestPossibleRegion().GetSize();
  ImageType::SizeType testSize;
  testSize = testReader->GetOutput()->GetLargestPossibleRegion().GetSize();

  if( baselineSize != testSize )
  {
    std::cerr << "The size of the Baseline image and Test image do not match!" << std::endl;
    std::cerr << "Baseline image: " << baselineImageFileName
              << " has size " << baselineSize << std::endl;
    std::cerr << "Test image:     " << testImageFileName
              << " has size " << testSize << std::endl;
    return EXIT_FAILURE;
  }

  // Now compare the two images
  typedef itk::Testing::ComparisonImageFilter< ImageType, ImageType > ComparisonFilterType;
  ComparisonFilterType::Pointer comparisonFilter = ComparisonFilterType::New();
  comparisonFilter->SetTestInput( testReader->GetOutput() );
  comparisonFilter->SetValidInput( baselineReader->GetOutput() );
  comparisonFilter->SetDifferenceThreshold( diffThreshold );
  try
  {
    comparisonFilter->Update();
  }
  catch( itk::ExceptionObject & err )
  {
    std::cerr << "Error during comparing image: " << err << std::endl;
    return EXIT_FAILURE;
  }

  //itk::SizeValueType numberOfDifferentPixels = comparisonFilter->GetNumberOfPixelsWithDifferences(); // in ITK4
  unsigned long numberOfDifferentPixels = comparisonFilter->GetNumberOfPixelsWithDifferences();

  if( numberOfDifferentPixels > 0 )
  {
    std::cerr << "There are " << numberOfDifferentPixels
              << " pixels with difference larger than "
              << allowedTolerance << "!" << std::endl;

    // Create name for diff image
    std::string diffImageFileName
                       = itksys::SystemTools::GetFilenamePath( testImageFileName );
    diffImageFileName += "/";
    diffImageFileName
                      += itksys::SystemTools::GetFilenameWithoutLastExtension( testImageFileName );
    diffImageFileName += "_DIFF";
    diffImageFileName += itksys::SystemTools::GetFilenameLastExtension( testImageFileName );

    typedef itk::ImageFileWriter< ImageType > WriterType;
    WriterType::Pointer writer = WriterType::New();
    writer->SetFileName( diffImageFileName );
    writer->SetInput( comparisonFilter->GetOutput() );
    try
    {
      writer->Write();
    }
    catch( itk::ExceptionObject & err )
    {
      std::cerr << "Error during writing difference image: " << err << std::endl;
      return EXIT_FAILURE;
    }

    if( numberOfDifferentPixels > allowedTolerance )
    {
      return EXIT_FAILURE;
    }

  } // end if discrepancies

  return EXIT_SUCCESS;

} // end main
