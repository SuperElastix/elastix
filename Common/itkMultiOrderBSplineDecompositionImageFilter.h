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
/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkMultiOrderBSplineDecompositionImageFilter.h,v $
  Language:  C++
  Date:      $Date: 2010-03-19 07:06:01 $
  Version:   $Revision: 1.12 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

  Portions of this code are covered under the VTK copyright.
  See VTKCopyright.txt or http://www.kitware.com/VTKCopyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkMultiOrderBSplineDecompositionImageFilter_h
#define __itkMultiOrderBSplineDecompositionImageFilter_h

#include <vector>

#include "itkImageLinearIteratorWithIndex.h"
#include "vnl/vnl_matrix.h"

#include "itkImageToImageFilter.h"

namespace itk
{
/** \class MultiOrderBSplineDecompositionImageFilter
 * \brief Calculates the B-Spline coefficients of an image.
 *        Spline order may be per dimension from 0 to 5 per.
 *
 * This class defines N-Dimension B-Spline transformation.
 * It is based on:
 *    [1] M. Unser,
 *       "Splines: A Perfect Fit for Signal and Image Processing,"
 *        IEEE Signal Processing Magazine, vol. 16, no. 6, pp. 22-38,
 *        November 1999.
 *    [2] M. Unser, A. Aldroubi and M. Eden,
 *        "B-Spline Signal Processing: Part I--Theory,"
 *        IEEE Transactions on Signal Processing, vol. 41, no. 2, pp. 821-832,
 *        February 1993.
 *    [3] M. Unser, A. Aldroubi and M. Eden,
 *        "B-Spline Signal Processing: Part II--Efficient Design and Applications,"
 *        IEEE Transactions on Signal Processing, vol. 41, no. 2, pp. 834-848,
 *        February 1993.
 * And code obtained from bigwww.epfl.ch by Philippe Thevenaz
 *
 * Limitations:  Spline order must be between 0 and 5.
 *               Spline order must be set before setting the image.
 *               Uses mirror boundary conditions.
 *               Can only process LargestPossibleRegion
 *
 * \sa itkBSplineInterpolateImageFunction
 *
 *  ***TODO: Is this an ImageFilter?  or does it belong to another group?
 * \ingroup ImageFilters
 * \ingroup SingleThreaded
 * \ingroup CannotBeStreamed
 */
template< class TInputImage, class TOutputImage >
class ITK_EXPORT MultiOrderBSplineDecompositionImageFilter :
  public         ImageToImageFilter< TInputImage, TOutputImage >
{
public:

  /** Standard class typedefs. */
  typedef MultiOrderBSplineDecompositionImageFilter       Self;
  typedef ImageToImageFilter< TInputImage, TOutputImage > Superclass;
  typedef SmartPointer< Self >                            Pointer;
  typedef SmartPointer< const Self >                      ConstPointer;

  /** Run-time type information (and related methods). */
  itkTypeMacro( MultiOrderBSplineDecompositionImageFilter, ImageToImageFilter );

  /** New macro for creation of through a Smart Pointer */
  itkNewMacro( Self );

  /** Inherit input and output image types from Superclass. */
  typedef typename Superclass::InputImageType         InputImageType;
  typedef typename Superclass::InputImagePointer      InputImagePointer;
  typedef typename Superclass::InputImageConstPointer InputImageConstPointer;
  typedef typename Superclass::OutputImagePointer     OutputImagePointer;

  typedef typename itk::NumericTraits< typename TOutputImage::PixelType >::RealType CoeffType;

  /** Dimension underlying input image. */
  itkStaticConstMacro( ImageDimension, unsigned int, TInputImage::ImageDimension );
  itkStaticConstMacro( OutputImageDimension, unsigned int,
    TOutputImage::ImageDimension );

  /** Iterator typedef support */
  typedef ImageLinearIteratorWithIndex< TOutputImage > OutputLinearIterator;

  /** Get/Sets the Spline Order, supports 0th - 5th order splines. The default
   *  is a 3rd order spline. */
  void SetSplineOrder( unsigned int order );

  void SetSplineOrder( unsigned int dimension, unsigned int order );

  void GetSplineOrder( unsigned int dimension )
  {
    return m_SplineOrder[ dimension ];
  }


  //itkGetMacro( SplineOrder, unsigned int * );

#ifdef ITK_USE_CONCEPT_CHECKING
  /** Begin concept checking */
  itkConceptMacro( DimensionCheck,
    ( Concept::SameDimension< ImageDimension, OutputImageDimension > ) );
  itkConceptMacro( InputConvertibleToOutputCheck,
    ( Concept::Convertible< typename TInputImage::PixelType,
    typename TOutputImage::PixelType > ) );
  itkConceptMacro( DoubleConvertibleToOutputCheck,
    ( Concept::Convertible< double, typename TOutputImage::PixelType > ) );
  /** End concept checking */
#endif

protected:

  MultiOrderBSplineDecompositionImageFilter();
  virtual ~MultiOrderBSplineDecompositionImageFilter() {}
  void PrintSelf( std::ostream & os, Indent indent ) const;

  void GenerateData();

  /** This filter requires all of the input image. */
  void GenerateInputRequestedRegion();

  /** This filter must produce all of its output at once. */
  void EnlargeOutputRequestedRegion( DataObject * output );

  /** These are needed by the smoothing spline routine. */
  std::vector< CoeffType > m_Scratch;             // temp storage for processing of Coefficients
  typename TInputImage::SizeType m_DataLength;    // Image size

  unsigned int m_SplineOrder[ ImageDimension ];            // User specified spline order per dimension (3rd or cubic is the default)
  double       m_SplinePoles[ 3 ];                         // Poles calculated for a given spline order
  int          m_NumberOfPoles;                            // number of poles
  double       m_Tolerance;                                // Tolerance used for determining initial causal coefficient
  unsigned int m_IteratorDirection;                        // Direction for iterator incrementing

private:

  MultiOrderBSplineDecompositionImageFilter( const Self & ); //purposely not implemented
  void operator=( const Self & );                            //purposely not implemented

  /** Determines the poles for dimension given the Spline Order. */
  virtual void SetPoles( unsigned int dimension );

  /** Converts a vector of data to a vector of Spline coefficients. */
  virtual bool DataToCoefficients1D();

  /** Converts an N-dimension image of data to an equivalent sized image
   *    of spline coefficients. */
  void DataToCoefficientsND();

  /** Determines the first coefficient for the causal filtering of the data. */
  virtual void SetInitialCausalCoefficient( double z );

  /** Determines the first coefficient for the anti-causal filtering of the data. */
  virtual void SetInitialAntiCausalCoefficient( double z );

  /** Used to initialize the Coefficients image before calculation. */
  void CopyImageToImage();

  /** Copies a vector of data from the Coefficients image to the m_Scratch vector. */
  void CopyCoefficientsToScratch( OutputLinearIterator & );

  /** Copies a vector of data from m_Scratch to the Coefficients image. */
  void CopyScratchToCoefficients( OutputLinearIterator & );

};

} // namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkMultiOrderBSplineDecompositionImageFilter.hxx"
#endif

#endif
