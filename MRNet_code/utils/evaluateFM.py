# -*- coding: utf-8 -*-
import numpy as np
import os
import PIL.Image as Image


def get_FM(outpath,gtpath):

    gtdir = gtpath
    outdir = outpath

    files = os.listdir(gtdir)
    eps = np.finfo(float).eps

    m_pres = np.zeros(21)
    m_recs = np.zeros(21)
    m_fms = np.zeros(21)
    m_thfm = 0
    m_mea = 0
    it = 1
    for i, name in enumerate(files):
        if not os.path.exists(gtdir + name):
            print(gtdir + name, 'does not exist')
        gt = Image.open(gtdir + name)
        gt = np.array(gt, dtype=np.uint8)


        mask=Image.open(outdir+name).convert('L')
        mask=mask.resize((np.shape(gt)[1],np.shape(gt)[0]))
        mask = np.array(mask, dtype=np.float)
        # salmap = cv2.resize(salmap,(W,H))

        if len(mask.shape) != 2:
            mask = mask[:, :, 0]
        mask = (mask - mask.min()) / (mask.max() - mask.min() + eps)
        gt[gt != 0] = 1
        pres = []
        recs = []
        fms = []
        mea = np.abs(gt-mask).mean()
        # threshold fm
        binary = np.zeros(mask.shape)
        th = 2*mask.mean()
        if th > 1:
            th = 1
        binary[mask >= th] = 1
        sb = (binary * gt).sum()
        pre = sb / (binary.sum()+eps)
        rec = sb / (gt.sum()+eps)
        thfm = 1.3 * pre * rec / (0.3 * pre + rec + eps)
        for th in np.linspace(0, 1, 21):
            binary = np.zeros(mask.shape)
            binary[ mask >= th] = 1
            pre = (binary * gt).sum() / (binary.sum()+eps)
            rec = (binary * gt).sum() / (gt.sum()+ eps)
            fm = 1.3 * pre * rec / (0.3*pre + rec + eps)
            pres.append(pre)
            recs.append(rec)
            fms.append(fm)
        fms = np.array(fms)
        pres = np.array(pres)
        recs = np.array(recs)
        m_mea = m_mea * (it-1) / it + mea / it
        m_fms = m_fms * (it - 1) / it + fms / it
        m_recs = m_recs * (it - 1) / it + recs / it
        m_pres = m_pres * (it - 1) / it + pres / it
        m_thfm = m_thfm * (it - 1) / it + thfm / it
        it += 1
    return m_thfm, m_mea

if __name__ == '__main__':
    m_thfm, m_mea=get_FM()
