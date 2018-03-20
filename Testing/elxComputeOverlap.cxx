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
 \brief Compute the overlap of two images.

 \verbinclude computeoverlap.help
 */
#include "itkCommandLineArgumentParser.h"

#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkAndImageFilter.h"
#include "itkImageRegionConstIterator.h"


/**
 * ******************* GetHelpString *******************
 */

std::string GetHelpString( void )
{
  std::stringstream ss;
  ss << "Usage:\n"
    << "elxComputeOverlap\n"
    << "This program computes the overlap of two images.\n"
    << "The results is computed as:\n"
    << "    2 * L1( im1 AND im2 )\n"
    << "  ------------------------\n"
    << "    L1( im1 ) + L1( im2 )\n\n"
    << "  -in      inputFilename1 inputFilename2";// << std::endl
//    << "Supported: 2D, 3D, (unsigned) char, (unsigned) short";
  return ss.str();

} // end GetHelpString()

//-------------------------------------------------------------------------------------

int main( int argc, char ** argv )
{
  /** Create a command line argument parser. */
  itk::CommandLineArgumentParser::Pointer parser = itk::CommandLineArgumentParser::New();
  parser->SetCommandLineArguments( argc, argv );
  parser->SetProgramHelpText( GetHelpString() );

  /** Get arguments. */
  std::vector<std::string> inputFileNames;
  bool retin = parser->GetCommandLineArgument( "-in", inputFileNames );

  parser->MarkArgumentAsRequired( "-in", "Two input filenames." );

  itk::CommandLineArgumentParser::ReturnValue validateArguments = parser->CheckForRequiredArguments();

  if( validateArguments == itk::CommandLineArgumentParser::FAILED )
  {
    return EXIT_FAILURE;
  }
  else if( validateArguments == itk::CommandLineArgumentParser::HELPREQUESTED )
  {
    return EXIT_SUCCESS;
  }

  //double tolerance = 1e-3;
  //parser->GetCommandLineArgument( "-tol", tolerance );

  /** Checks. */
  if( !retin || inputFileNames.size() != 2 )
  {
    std::cerr << "ERROR: You should specify two input file names with \"-in\"." << std::endl;
    return EXIT_FAILURE;
  }

  /** Hard-coded image dimension and pixel type. */
  const unsigned int Dimension = 3;
  typedef unsigned char PixelType;

  /** Typedefs. */
  typedef itk::Image< PixelType, Dimension >          ImageType;
  typedef itk::ImageFileReader< ImageType >           ImageReaderType;
  typedef ImageReaderType::Pointer                    ImageReaderPointer;
  typedef itk::AndImageFilter<
    ImageType, ImageType, ImageType >                 AndFilterType;
  typedef AndFilterType::Pointer                      AndFilterPointer;
  typedef itk::ImageRegionConstIterator< ImageType >  IteratorType;

  /** Create readers and an AND filter. */
  ImageReaderPointer reader1 = ImageReaderType::New();
  reader1->SetFileName( inputFileNames[ 0 ].c_str() );
  ImageReaderPointer reader2 = ImageReaderType::New();
  reader2->SetFileName( inputFileNames[ 1 ].c_str() );
  AndFilterPointer ANDFilter = AndFilterType::New();
  ANDFilter->SetInput1( reader2->GetOutput() );
  ANDFilter->SetInput2( reader1->GetOutput() );
  //finalANDFilter->SetCoordinateTolerance( tolerance );
  //finalANDFilter->SetDirectionTolerance( tolerance );

  /** Do the AND operation. */
  try
  {
    ANDFilter->Update();
  }
  catch( itk::ExceptionObject & excp )
  {
    std::cerr << excp << std::endl;
    return EXIT_FAILURE;
  }

  /** Now calculate the L1-norms. */

  /** Create iterators. */
  IteratorType iteratorA( ANDFilter->GetInput( 1 ),
    ANDFilter->GetInput( 1 )->GetLargestPossibleRegion() );
  IteratorType iteratorB( ANDFilter->GetInput( 0 ),
    ANDFilter->GetInput( 0 )->GetLargestPossibleRegion() );
  IteratorType iteratorC( ANDFilter->GetOutput(),
    ANDFilter->GetOutput()->GetLargestPossibleRegion() );

  /** Determine size of first object. */
  long long sumA = 0;
  for( iteratorA.GoToBegin(); !iteratorA.IsAtEnd(); ++iteratorA )
  {
    if( iteratorA.Value() )
    {
      ++sumA;
    }
  }
  //std::cout << "Size of first object: " << sumA << std::endl;

  /** Determine size of second object. */
  long long sumB = 0;
  for( iteratorB.GoToBegin(); !iteratorB.IsAtEnd(); ++iteratorB )
  {
    if( iteratorB.Value() )
    {
      ++sumB;
    }
  }
  //std::cout << "Size of second object: " << sumB << std::endl;

  /** Determine size of cross-section. */
  long long sumC = 0;
  for( iteratorC.GoToBegin(); !iteratorC.IsAtEnd(); ++iteratorC )
  {
    if( iteratorC.Value() )
    {
      ++sumC;
    }
  }
  //std::cout << "Size of cross-section of both objects: " << sumC << std::endl;

  /** Calculate the overlap. */
  double overlap;
  if( ( sumA + sumB ) == 0 )
  {
    overlap = 0;
  }
  else
  {
    overlap = static_cast<double>( 2 * sumC ) / static_cast<double>( sumA + sumB );
  }

  /** Format the output and show overlap. */
  std::cout << std::fixed << std::showpoint;
  std::cout << "Overlap: " << overlap << std::endl;

  /** End program. */
  return EXIT_SUCCESS;

} // end main
