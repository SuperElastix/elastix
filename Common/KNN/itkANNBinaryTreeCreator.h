/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __itkANNBinaryTreeCreator_h
#define __itkANNBinaryTreeCreator_h

#include "itkObject.h"
#include "itkObjectFactory.h"
#include "ANN/ANN.h"

namespace itk
{

/**
 * \class ANNBinaryTreeCreator
 *
 * \brief
 *
 *
 * \ingroup ANNwrap
 */

class ANNBinaryTreeCreator : public Object
{
public:

  /** Standard itk. */
  typedef ANNBinaryTreeCreator       Self;
  typedef Object                     Superclass;
  typedef SmartPointer< Self >       Pointer;
  typedef SmartPointer< const Self > ConstPointer;

  /** New method for creating an object using a factory. */
  itkNewMacro( Self );

  /** ITK type info. */
  itkTypeMacro( ANNBinaryTreeCreator, Object );

  /** ANN typedef's. */
  //typedef ANNpointSet     ANNTreeBaseType;
  typedef ANNkd_tree    ANNkDTreeType;
  typedef ANNbd_tree    ANNbdTreeType;
  typedef ANNbruteForce ANNBruteForceTreeType;
  typedef ANNpointArray ANNPointArrayType;
  typedef ANNsplitRule  ANNSplitRuleType;
  typedef ANNshrinkRule ANNShrinkRuleType;

  /** Static funtions to create and delete ANN trees.
   * We keep a reference count so that when no more trees
   * of any sort exist, we can call annClose(). This little
   * function is cause of going through the trouble of creating
   * this class with static creating functions.
   */

  /** Static function to create an ANN kDTree. */
  static ANNkDTreeType * CreateANNkDTree( ANNPointArrayType pa, int n, int d, int bs = 1,
    ANNSplitRuleType split = ANN_KD_SUGGEST );

  /** Static function to create an ANN bdTree. */
  static ANNbdTreeType * CreateANNbdTree( ANNPointArrayType pa, int n, int d, int bs = 1,
    ANNSplitRuleType split = ANN_KD_SUGGEST, ANNShrinkRuleType shrink = ANN_BD_SUGGEST );

  /** Static function to create an ANN BruteForceTree. */
  static ANNBruteForceTreeType * CreateANNBruteForceTree( ANNPointArrayType pa, int n, int d );

  /** Static function to delete any ANN tree that inherits from kDTree (not brute force). */
  static void DeleteANNkDTree( ANNkDTreeType * & tree );

  /** Static function to delete an ANN BruteForceTree. */
  static void DeleteANNBruteForceTree( ANNBruteForceTreeType * & tree );

  /** Static function to increase the reference count to ANN trees. */
  static void IncreaseReferenceCount( void );

  /** Static function to decrease the reference count to ANN trees. */
  static void DecreaseReferenceCount( void );

protected:

  ANNBinaryTreeCreator(){}
  virtual ~ANNBinaryTreeCreator(){}

private:

  ANNBinaryTreeCreator( const Self & );   // purposely not implemented
  void operator=( const Self & );         // purposely not implemented

  /** Member variables. */
  static unsigned int m_NumberOfANNBinaryTrees;

};

} // end namespace itk

#endif // end #ifndef __itkANNBinaryTreeCreator_h
