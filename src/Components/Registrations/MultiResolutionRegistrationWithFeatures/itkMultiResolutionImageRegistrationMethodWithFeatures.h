/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkMultiResolutionImageRegistrationMethodWithFeatures_h
#define __itkMultiResolutionImageRegistrationMethodWithFeatures_h

#include "itkMultiInputMultiResolutionImageRegistrationMethodBase.h"


namespace itk
{

/** \class MultiResolutionImageRegistrationMethodWithFeatures
 * \brief Class for multi-resolution image registration methods
 *
 * This class is an extension of the itk class
 * MultiResolutionImageRegistrationMethod. It allows the use
 * of multiple metrics, which are summed, multiple images,
 * multiple interpolators, and/or multiple image pyramids.
 *
 * Make sure the following is true:\n
 *   nrofmetrics >= nrofinterpolators >= nrofmovingpyramids >= nrofmovingimages\n
 *   nrofmetrics >= nroffixedpyramids >= nroffixedimages\n
 *   nroffixedregions == nroffixedimages\n
 *
 *   nrofinterpolators == nrofmetrics OR nrofinterpolators == 1\n
 *   nroffixedimages == nrofmetrics OR nroffixedimages == 1\n
 *   etc...
 *
 * You may also set an interpolator/fixedimage/etc to NULL, if you
 * happen to know that the corresponding metric is not an
 * ImageToImageMetric, but a regularizer for example (which does
 * not need an image.
 *
 *
 * \sa ImageRegistrationMethod
 * \sa MultiResolutionImageRegistrationMethod
 * \ingroup RegistrationFilters
 */

template <typename TFixedImage, typename TMovingImage>
class MultiResolutionImageRegistrationMethodWithFeatures :
  public MultiInputMultiResolutionImageRegistrationMethodBase<TFixedImage, TMovingImage>
{
public:
  /** Standard class typedefs. */
  typedef MultiResolutionImageRegistrationMethodWithFeatures Self;
  typedef MultiInputMultiResolutionImageRegistrationMethodBase<
    TFixedImage, TMovingImage>                               Superclass;
  typedef SmartPointer<Self>                                 Pointer;
  typedef SmartPointer<const Self>                           ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( MultiResolutionImageRegistrationMethodWithFeatures,
    MultiInputMultiResolutionImageRegistrationMethodBase );

  /**  Superclass types */
  typedef typename Superclass::FixedImageType           FixedImageType;
  typedef typename Superclass::FixedImageConstPointer   FixedImageConstPointer;
  typedef typename Superclass::FixedImageRegionType     FixedImageRegionType;
  typedef typename Superclass::FixedImageRegionPyramidType  FixedImageRegionPyramidType;
  typedef typename Superclass::MovingImageType          MovingImageType;
  typedef typename Superclass::MovingImageConstPointer  MovingImageConstPointer;

  typedef typename Superclass::MetricType               MetricType;
  typedef typename Superclass::MetricPointer            MetricPointer;
  typedef typename Superclass::TransformType            TransformType;
  typedef typename Superclass::TransformPointer         TransformPointer;
  typedef typename Superclass::InterpolatorType         InterpolatorType;
  typedef typename Superclass::InterpolatorPointer      InterpolatorPointer;
  typedef typename Superclass::OptimizerType            OptimizerType;
  typedef typename OptimizerType::Pointer               OptimizerPointer;
  typedef typename Superclass::FixedImagePyramidType    FixedImagePyramidType;
  typedef typename Superclass::FixedImagePyramidPointer FixedImagePyramidPointer;
  typedef typename Superclass::MovingImagePyramidType   MovingImagePyramidType;
  typedef typename
    Superclass::MovingImagePyramidPointer               MovingImagePyramidPointer;

  typedef typename Superclass::TransformOutputType      TransformOutputType;
  typedef typename Superclass::TransformOutputPointer   TransformOutputPointer;
  typedef typename
    Superclass::TransformOutputConstPointer             TransformOutputConstPointer;

  typedef typename Superclass::ParametersType           ParametersType;
  typedef typename Superclass::DataObjectPointer        DataObjectPointer;

protected:

  /** Constructor. */
  MultiResolutionImageRegistrationMethodWithFeatures(){};

  /** Destructor. */
  virtual ~MultiResolutionImageRegistrationMethodWithFeatures() {};

  /** Function called by PreparePyramids, which checks if the user input
   * regarding the image pyramids is ok.
   */
  virtual void CheckPyramids( void ) throw (ExceptionObject);

private:
  MultiResolutionImageRegistrationMethodWithFeatures(const Self&); // purposely not implemented
  void operator=(const Self&); // purposely not implemented

}; // end class MultiResolutionImageRegistrationMethodWithFeatures


} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkMultiResolutionImageRegistrationMethodWithFeatures.hxx"
#endif

#endif // end #ifndef __itkMultiResolutionImageRegistrationMethodWithFeatures_h
